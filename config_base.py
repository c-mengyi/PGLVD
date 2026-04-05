data_config = {
    "CASIA": {
        "image_directory": "/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/cropped_images",
        "known_list_path":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/pkl/CASIA_known_list.pkl",
        "unknown_list_path":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/pkl/CASIA_unknown_list.pkl",
    },
    "IJBC":{
        "image_directory" : "/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/IJBC/cropped_images",
        "ijbc_t_m": "/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/IJBC/meta/ijbc_face_tid_mid.txt",
        "ijbc_5pts":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/IJBC/meta/ijbc_name_5pts_score.txt",
        "ijbc_gallery_1":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/IJBC/meta/ijbc_1N_gallery_G1.csv",
        "ijbc_gallery_2":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/IJBC/meta/ijbc_1N_gallery_G2.csv",
        "ijbc_probe":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/IJBC/meta/ijbc_1N_probe_mixed.csv",
        "processed_img_root":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/IJBC/cropped_images",
        "plk_file_root":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/IJBC/data"
    },
    "VGGFACE": {
            "image_directory": "/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/vggface2_mtcnn_160",
            "known_list_path":"/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/pkl/selected_vgg_subfolder_names.pkl",
        },
}

encoder_config = {
    "VGG19": "/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/VGG19_CosFace.chkpt",
    "Res50": "/share/home/ncu_418000230018/face/OSFI-by-FineTuning-main418ncu/ResIR50_CosFace.chkpt",
}