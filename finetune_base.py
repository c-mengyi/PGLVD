import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def hflip(x):
    """水平翻转，x: [B, C, H, W]"""
    return torch.flip(x, dims=[3])


def forward_head(args, classifier, feat, label):
    """
    兼容普通 CosFace / MagFace 头
    """
    if getattr(args, "head_type", "") == "mag":
        logit, sim, loss_g = classifier(feat, label)
        return logit, sim, loss_g
    else:
        logit, sim = classifier(feat, label)
        return logit, sim, 0.0


def extract_global_feature(encoder, img):
    """
    公式 (3.4):
    f_G(x) = (f(x) + f(flip(x))) / 2
    """
    feat = encoder(img)
    feat_flip = encoder(hflip(img))
    feat_g = 0.5 * (feat + feat_flip)
    return feat_g


@torch.no_grad()
def weight_imprinting(args, encoder, galleryloader, num_cls, feat_dim=512):
    """
    公式 (3.4) + (3.5)
    使用图库样本计算类别原型:
    P_y = sum(f_G(x)) / ||sum(f_G(x))||_2
    """
    encoder.eval()

    proto_sum = torch.zeros(num_cls, feat_dim, device=args.device)
    cls_count = torch.zeros(num_cls, device=args.device)

    for img, label in galleryloader:
        img = img.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True).long()

        feat_g = extract_global_feature(encoder, img)  # [B, D]
        proto_sum.index_add_(0, label, feat_g)
        cls_count.index_add_(0, label, torch.ones_like(label, dtype=torch.float))

    valid_mask = cls_count > 0
    if valid_mask.any():
        proto_sum[valid_mask] = F.normalize(proto_sum[valid_mask], p=2, dim=1)

    return proto_sum  # [num_cls, feat_dim]


def extract_local_features(args, encoder, img):
    """
    公式 (3.6) + (3.7)
    将输入图像沿垂直方向划分为 J 个子区域，
    再恢复到原尺寸后提取局部特征。
    返回: [B, J, D]
    """
    b, c, h, w = img.shape
    local_feats = []

    for j in range(args.J):
        # 沿高度方向切分为 J 个水平条带
        patch = img[:, :, j * h // args.J:(j + 1) * h // args.J, :]  # [B, C, h/J, W]
        patch = F.interpolate(
            patch, size=(h, w), mode="bilinear", align_corners=False
        )
        feat_l = encoder(patch)  # [B, D]
        local_feats.append(feat_l)

    local_feats = torch.stack(local_feats, dim=1)  # [B, J, D]
    return local_feats


def init_classifier_from_prototypes(classifier, prototypes):
    """
    用原型初始化分类器权重
    兼容两种常见权重形状:
    1) [num_cls, feat_dim]
    2) [feat_dim, num_cls]
    """
    with torch.no_grad():
        if classifier.weight.shape == prototypes.shape:
            classifier.weight.copy_(prototypes)
        elif classifier.weight.shape == prototypes.t().shape:
            classifier.weight.copy_(prototypes.t())
        else:
            raise ValueError(
                f"classifier.weight shape = {classifier.weight.shape}, "
                f"but prototypes shape = {prototypes.shape}"
            )


def prototype_guided_expansion(global_feat, local_feat, prototypes, K, alpha, tau, th):
    """
    公式 (3.8) - (3.11)

    global_feat: [B, D]
    local_feat : [B, J, D]
    prototypes : [C, D]

    返回:
        exp_feat : [N_exp, D]
        exp_label: [N_exp]
    """
    K = min(K, prototypes.size(0))

    # 归一化后计算相似度
    proto_norm = F.normalize(prototypes, p=2, dim=1)      # [C, D]
    global_norm = F.normalize(global_feat, p=2, dim=1)    # [B, D]
    local_norm = F.normalize(local_feat, p=2, dim=2)      # [B, J, D]

    # 公式 (3.8): 选择前 K 个最近邻原型
    sim_gp = torch.matmul(global_norm, proto_norm.t())    # [B, C]
    topk_idx = torch.topk(sim_gp, k=K, dim=1).indices     # [B, K]
    candidate_proto = prototypes[topk_idx]                # [B, K, D]
    candidate_proto_norm = proto_norm[topk_idx]           # [B, K, D]

    # 公式 (3.9): 计算语义一致性权重
    # sim(local_feat[b, j], candidate_proto[b, k])
    sim_lp = torch.einsum("bjd,bkd->bjk", local_norm, candidate_proto_norm)  # [B, J, K]
    weights = F.softmax(tau * sim_lp, dim=1)                                  # 对 J 做 softmax

    # fatt(i,k): 对局部特征加权融合
    f_att = torch.einsum("bjk,bjd->bkd", weights, local_feat)  # [B, K, D]

    # 公式 (3.10): f_new(i,k) = alpha * P_nk + (1 - alpha) * f_att(i,k)
    f_new = alpha * candidate_proto + (1.0 - alpha) * f_att   # [B, K, D]

    # 公式 (3.11): 置信度筛选
    f_new_norm = F.normalize(f_new, p=2, dim=2)
    sim_all = torch.einsum("bkd,cd->bkc", f_new_norm, proto_norm)  # [B, K, C]

    # 与目标原型的相似度
    target_sim = sim_all.gather(2, topk_idx.unsqueeze(-1)).squeeze(-1)  # [B, K]

    # 与非目标原型的最大相似度
    other_sim = sim_all.clone()
    other_sim.scatter_(2, topk_idx.unsqueeze(-1), float("-inf"))
    max_other_sim = other_sim.max(dim=2).values  # [B, K]

    # 两个条件同时满足:
    # 1) sim(f_new, P_nk) > th
    # 2) sim(f_new, P_nk) - max_{c!=nk} sim(f_new, P_c) > 0
    keep_mask = (target_sim > th) & ((target_sim - max_other_sim) > 0)

    exp_feat = f_new[keep_mask]              # [N_exp, D]
    exp_label = topk_idx[keep_mask].long()   # [N_exp]

    return exp_feat, exp_label


def fine_tune(args,
              galleryloader,
              trainloader,
              num_cls,
              encoder,
              classifier,
              optimizer,
              scheduler=None,
              verbose=True):
    """
    两阶段微调:
    - 总共 20 个 epoch
    - 前 round(20 * T_L) 个 epoch: 阶段2（初始微调）
    - 后 20 - round(20 * T_L) 个 epoch: 阶段3（再微调）

    参数说明:
    - galleryloader: 用于计算原型的图库 loader
    - trainloader  : 用于训练的 loader
      如果图库和训练集相同，可以两者传同一个 loader
    """
    ce_loss = nn.CrossEntropyLoss()

    total_epochs = 20
    stage2_epochs = int(round(total_epochs * args.T_L))
    stage2_epochs = max(0, min(total_epochs, stage2_epochs))
    stage3_epochs = total_epochs - stage2_epochs

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs
        )

    use_amp = str(args.device).startswith("cuda")

    log_dir = os.path.join("logs", f"{classifier.__class__.__name__}_two_stage")
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Total epochs = {total_epochs}")
    print(f"Stage 2 epochs = {stage2_epochs}")
    print(f"Stage 3 epochs = {stage3_epochs}")

    for epoch in range(total_epochs):
        # 阶段标识
        if epoch < stage2_epochs:
            stage_name = "stage2"   # 初始微调
        else:
            stage_name = "stage3"   # 再微调

        # 每个 epoch 重新计算原型，并用其初始化分类器权重
        prototypes = weight_imprinting(
            args=args,
            encoder=encoder,
            galleryloader=galleryloader,
            num_cls=num_cls,
            feat_dim=512
        )
        init_classifier_from_prototypes(classifier, prototypes)

        encoder.train()
        classifier.train()

        train_corr, train_tot = 0, 0
        loss_sum = 0.0

        for img, label in trainloader:
            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True).long()

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 全局特征（带翻转增强）
                global_feat = extract_global_feature(encoder, img)  # [B, D]

                if stage_name == "stage2":
                    # ===== 阶段2：初始微调 =====
                    all_feat = global_feat
                    all_label = label

                else:
                    # ===== 阶段3：再微调 =====
                    # 1. 提取局部特征
                    local_feat = extract_local_features(args, encoder, img)  # [B, J, D]

                    # 2. 原型引导特征扩展 + 置信度筛选
                    exp_feat, exp_label = prototype_guided_expansion(
                        global_feat=global_feat,
                        local_feat=local_feat,
                        prototypes=prototypes.detach(),  # 原型仅作引导
                        K=args.c_k,
                        alpha=args.alpha,
                        tau=args.tau,
                        th=args.th
                    )

                    # 3. 与原始全局特征联合训练
                    if exp_feat.numel() > 0:
                        all_feat = torch.cat([global_feat, exp_feat], dim=0)
                        all_label = torch.cat([label, exp_label], dim=0)
                    else:
                        all_feat = global_feat
                        all_label = label

                # CosFace / MagFace 前向
                logit, sim, extra_loss = forward_head(args, classifier, all_feat, all_label)
                loss = ce_loss(logit, all_label) + extra_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(sim, dim=1)
            train_corr += pred.eq(all_label).sum().item()
            train_tot += all_label.size(0)
            loss_sum += loss.item()

        scheduler.step()

        train_acc = 100.0 * train_corr / max(train_tot, 1)
        avg_loss = loss_sum / max(len(trainloader), 1)

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("train/lr", get_lr(optimizer), epoch)

        if verbose:
            print(
                f"[{stage_name}] epoch {epoch + 1:02d}/{total_epochs} | "
                f"loss: {avg_loss:.4f} | acc: {train_acc:.2f}% | lr: {get_lr(optimizer):.2e}"
            )

    writer.close()
