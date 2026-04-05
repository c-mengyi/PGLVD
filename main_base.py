import config_base as config
from dataset_base import open_set_folds, face_dataset, ijbc_dataset, partition_dataset
from model_base import fetch_encoder, head
from finetune_base import weight_imprinting, fine_tune
from utils_base import save_dir_far_curve, save_dir_far_excel

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import json
import pprint
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import datetime


# for boolean parser argument
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("'True' or 'False' expected")

def False_or_float(v):
    if v == "False":
        return False
    else:
        return float(v)

parser = argparse.ArgumentParser()
# basic arguments
parser.add_argument("--device_id", type=int, default=3)
parser.add_argument("--lr",default=1e-3,type=float)

parser.add_argument("--batch_size",default=128,type=int)
parser.add_argument("--num_epochs",default=20,type=int,help="num_epochs for fine-tuning")

# dataset arguments
parser.add_argument("--dataset", type=str, default='IJBC', help="['CASIA','IJBC']")
parser.add_argument("--probe_dataset", type=str, default='probe', help="['probe','val']")
parser.add_argument("--num_gallery", type=int, default=3, help="number of gallery images per identity")

# encoder arguments
parser.add_argument("--encoder", type=str, default='Res50', help="['VGG19','Res50']")
parser.add_argument("--head_type", type=str, default='cos', help="['arc', 'cos', 'mag','norm','softmax']")

# main arguments: classifier init / finetune layers / matcher
parser.add_argument("--classifier_init", type=str, default='WI',
                    help="['Random','LP','WI','None']")  # Random Init. / Linear Probing / Weight Imprinting
parser.add_argument("--finetune_layers", type=str, default='BN',
                    help="['None','Full','Partial','PA','BN']")  # 'None' refers to no fine-tuning
parser.add_argument("--matcher", type=str, default='NAC',
                    help="['org','NAC','EVM']")  # unused argument: refer to the results

# misc. arguments: no need to change
parser.add_argument("--arc_s",default=32,type=float,help="scale for ArcFace")
parser.add_argument("--arc_m",default=0.4,type=float,help="margin for ArcFace")
parser.add_argument("--cos_m",default=0.4,type=float,help="margin for CosFace")
parser.add_argument("--train_output",type=str2bool,default=False,
                    help="if True, train output layer")

parser.add_argument("--k",default=16,type=int,help="k for NAC")

#PGLVD
parser.add_argument("--T_L", default=0.5, type=float,
                    help="ratio of stage-2 epochs in total 20 epochs")

parser.add_argument("--J", default=4, type=int,
                    help="number of local regions split along vertical direction")

parser.add_argument("--c_k", default=30, type=int,
                    help="top-K nearest prototypes for prototype-guided expansion")

parser.add_argument("--alpha", default=0.9, type=float,
                    help="fusion weight between prototype and local aggregated feature")

parser.add_argument("--tau", default=10.0, type=float,
                    help="temperature scaling factor for local-prototype weighting")

parser.add_argument("--th", default=0.5, type=float,
                    help="similarity threshold for filtering expanded features")

args = parser.parse_args()



def main(args):
    # check arguments
    global classifier
    assert args.classifier_init in ['Random','LP','WI','None'], 'classifier_init must be one of ["Random","LP","WI",' \
                                                                '"None"]'
    assert args.finetune_layers in ['None','Full','Partial','PA','BN'], \
        "finetune_layers must be one of ['None','Full','Partial','PA','BN']"

    # fix random seed
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # set device
    args.device = torch.device(args.device_id)
    if args.finetune_layers == 'None':
        exp_name = 'Pretrained'
    else:
        exp_name = f'{args.classifier_init}_{args.finetune_layers}/{args.head_type}'
    # 获取当前时间
    current_time = datetime.datetime.now()
    # 将时间格式化为字符串，例如：2022-01-01_12-30-45
    time_str = current_time.strftime("%Y-%m-%d__%H:%M:%S")
    save_dir = f"results/{args.dataset}_{args.encoder}_{args.probe_dataset}/{exp_name}/{time_str}/"
    os.makedirs(save_dir, exist_ok=True)
    print("results are saved at: ", save_dir)

    # save arguments
    argdict = args.__dict__.copy()
    argdict['device'] = argdict['device'].type + f":{argdict['device'].index}"
    with open(save_dir + '/args.txt', 'w') as fp:
        json.dump(argdict, fp, indent=2)


    train_trf = transforms.Compose([
        transforms.RandomResizedCrop(size=112, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])
    eval_trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])
    """
    prepare G, K, U sets for evaluation
    increase batch_size for faster inference
    """
    # prepare dataset: config
    data_config = config.data_config[args.dataset]
    if args.dataset == 'CASIA':
        folds = open_set_folds(data_config["image_directory"], data_config["known_list_path"],
                               data_config["unknown_list_path"], args.num_gallery)
        dataset_val = face_dataset(folds.val, eval_trf, img_size=112)
        dataset_gallery = face_dataset(folds.G, eval_trf, img_size=112)
        dataset_probe = face_dataset(folds.test, eval_trf, img_size=112)
        data_loader_gallery = DataLoader(dataset_gallery, batch_size=256, shuffle=False, num_workers=4)
        data_loader_probe = DataLoader(dataset_probe, batch_size=256, shuffle=False, num_workers=4)
        data_loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=4)
        num_cls = folds.num_known
        trainset_gallery = face_dataset(folds.G, train_trf, 112)
    if args.dataset == 'IJBC':
        Gallery, Known, Unknown, Probe, Val, Test, num_cls = partition_dataset(data_config["ijbc_t_m"],
                                                                               data_config["ijbc_5pts"],
                                                                               data_config["ijbc_gallery_1"],
                                                                               data_config["ijbc_gallery_2"],
                                                                               data_config["ijbc_probe"],
                                                                               data_config["processed_img_root"],
                                                                               data_config["plk_file_root"],
                                                                               args.num_gallery)
        dataset_gallery = ijbc_dataset(Gallery, eval_trf, img_size=112)
        data_loader_gallery = DataLoader(dataset_gallery, batch_size=256, shuffle=False, num_workers=4)
        dataset_probe = ijbc_dataset(Test, eval_trf, img_size=112)
        data_loader_probe = DataLoader(dataset_probe, batch_size=256, shuffle=False, num_workers=4)
        dataset_val = ijbc_dataset(Val, eval_trf, img_size=112)
        data_loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=4)
        trainset_gallery = ijbc_dataset(Gallery, train_trf, 112)

    '''
    prepare encoder
    '''
    encoder = fetch_encoder.fetch(args.device, config.encoder_config,
                                  args.encoder, args.finetune_layers, args.train_output)

    '''
    fine-tune
    '''
    if args.finetune_layers != "None":  # for 'None', no fine-tuning is done
        if args.head_type == "arc":
            classifier = head.arcface_head(args.device, 512, num_cls, s=args.arc_s, m=args.arc_m, use_amp=True)
        elif args.head_type == "cos":
            classifier = head.cosface_head(512, num_cls, s=args.arc_s, m=args.cos_m)
        elif args.head_type == "mag":
            classifier = head.magface_head(args.device, 512, num_cls, s=args.arc_s, use_amp=True)
        elif args.head_type == "norm":
            classifier = head.normface_head(512, num_cls, s=args.arc_s)
        elif args.head_type == "softmax":
            classifier = head.softmax_head(512, num_cls)
        classifier.to(args.device)
        # classifier initialization
        if args.classifier_init == 'WI':
            prototypes = weight_imprinting(args, encoder, data_loader_gallery, num_cls, 512)
            classifier.weight = nn.Parameter(prototypes.T)
        else:
            pass  # just use random weights for classifier

        # set optimizer & LR scheduler
        optimizer = optim.Adam([{"params": encoder.parameters(), "lr": args.lr},
                                {"params": classifier.parameters(), "lr": args.lr}],
                               weight_decay=1e-3)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

        train_loader_gallery = DataLoader(trainset_gallery, batch_size=args.batch_size, shuffle=True, num_workers=4)
        # load training dataset (random shuffling & augmentation)
        fine_tune(args, train_loader_gallery,train_loader_gallery, num_cls,encoder, classifier, optimizer, scheduler, verbose=True)
    '''
    evaluate encoder
    '''
    encoder.eval()
    flip = transforms.RandomHorizontalFlip(p=1)
    Gfeat = torch.FloatTensor([]).to(args.device)
    Glabel = torch.LongTensor([])
    for img, label in tqdm(data_loader_gallery):
        img = img.to(args.device)
        with torch.no_grad():
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
        Gfeat = torch.cat((Gfeat, feat), dim=0)
        Glabel = torch.cat((Glabel, label), dim=0)
    Pfeat = torch.FloatTensor([]).to(args.device)
    Plabel = torch.LongTensor([])
    if args.probe_dataset == "probe":
        data_loader = data_loader_probe
    if args.probe_dataset == "val":
        data_loader = data_loader_val

    for img, label in tqdm(data_loader):
        img = img.to(args.device)
        with torch.no_grad():
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
        Pfeat = torch.cat((Pfeat, feat), dim=0)
        Plabel = torch.cat((Plabel, label), dim=0)
    Gfeat = Gfeat.cpu()
    Pfeat = Pfeat.cpu()
    # save results
    save_dir_far_curve(Gfeat, Glabel, Pfeat, Plabel, save_dir)
    save_dir_far_excel(Gfeat, Glabel, Pfeat, Plabel, save_dir)


if __name__ == '__main__':
    pprint.pprint(vars(args))
    main(args)