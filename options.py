import argparse
from torchvision import transforms

# -------------------------- Hyperparameters ------------------------------
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_LOAD_PATH', default="./model_result_full/epoch_42_loss_0.5287_Img_loss_0.0114_LMK_loss0.5174_Recog_loss0.0019.pth", type=str, help='saved model path')
# MODEL_LOAD_PATH="./model_result_full/epoch_04_loss_15.5574_lmk_loss_0.0120_img_loss0.7773.pth"
    parser.add_argument('--CACDDataset_path', default='./CACD2000', type=str, help='CACDDataset path')
    parser.add_argument('--Newdataset_path', default='./data', type=str, help='CACDDataset path')
    parser.add_argument('--bfm_folder', type=str, default='./BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    parser.add_argument('--MODEL_SAVE_PATH', default='model_result_full', type=str, help='model save path')
    parser.add_argument('--RESULT_SAVE_PATH', default='result_full', type=str, help='result save path')
    parser.add_argument('--video_rate', default=2, type=int, help='output video frame rate')
    parser.add_argument('--video_start_epoch', default=0, type=int, help='output video start epoch image')

    
    parser.add_argument('--BATCH_SIZE', default=16, type=int, help='batch size')
    parser.add_argument('--NUM_EPOCH', default=20, type=int, help='epochh number')
    parser.add_argument('--VERBOSE_STEP', default=50, type=int, help='progress printing step number')
    parser.add_argument('--seed', default=0, type=int, help='torch seed')
    parser.add_argument('--LR', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--feat_method', default='method1', type=str, choices=['method1', 'method2'], help='choose type of feat loss method')
    
    parser.add_argument('--VIS_BATCH_IDX', default=7, type=int, help='visualization batch index')
    parser.add_argument('--VIS_COL_NUM', default=8, type=int, help='visualization image column number')
    
    parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
    parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
    parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
    parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
    parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
    parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')
    parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
    parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
    parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
    parser.add_argument('--accimage', action='store_true', help='If true, accimage is used to load images.')
    parser.add_argument('--camera_distance', default=2.732, type=float, help='camera distance')
    parser.add_argument('--elevation', default=0, type=float, help='elevation')
    parser.add_argument('--azimuth', default=0, type=float, help='azimuth')
    args = parser.parse_args()
    args.train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    args.val_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    args.inv_normalize = transforms.Compose([
                        transforms.Normalize(
                                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                    std=[1/0.229, 1/0.224, 1/0.255])
        ])
    return args

# BATCH_SIZE=2
# NUM_EPOCH=50
# VERBOSE_STEP=50
# LR=3e-5
# VIS_BATCH_IDX=7
# LMK_LOSS_WEIGHT=1
# RECOG_LOSS_WEIGHT=20
# MODEL_LOAD_PATH=None
# SEED=0