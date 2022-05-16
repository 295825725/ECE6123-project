import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from tqdm import tqdm
from skimage import io
from options import parse_opts
import soft_renderer as sr
from dataset import CACDDataset
from model import BaseModel
from loss import BFMFaceLoss
import glob
import subprocess

# ------------------------- plot visualization --------------------------
def visualize_person(col_num, gt_imgs, recon_imgs):
    num_face = len(gt_imgs)
    num_cols = col_num
    num_rows = int(num_face/num_cols)
    if num_rows==0:
        num_cols=num_face
        num_rows=1
    canvas = np.zeros((num_rows*224, num_cols*224*2, 3))
    img_idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            gt_img = gt_imgs[img_idx].cpu()
            recon_img = recon_imgs[img_idx].cpu()
            gt_img = gt_img.permute(1,2,0).cpu().numpy()
            recon_img = recon_img[:3,:,:].permute(1,2,0).numpy()
            canvas[i*224:(i+1)*224, j*224*2:(j+1)*224*2-224, :3] = gt_img
            canvas[i*224:(i+1)*224, j*224*2+224:(j+1)*224*2, :4] = recon_img
            img_idx += 1
    return (np.clip(canvas,0,1)*255).astype(np.uint8)

def visualize_batch(col_num, gt_imgs, recon_imgs):
    gt_imgs = gt_imgs.cpu()
    recon_imgs = recon_imgs.cpu()
    bs = gt_imgs.shape[0]
    num_cols = col_num
    num_rows = int(bs/num_cols)
    canvas = np.zeros((num_rows*224, num_cols*224*2, 3))
    img_idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            gt_img = gt_imgs[img_idx].permute(1,2,0).numpy()
            recon_img = recon_imgs[img_idx,:3,:,:].permute(1,2,0).numpy()
            canvas[i*224:(i+1)*224, j*224*2:(j+1)*224*2-224, :3] = gt_img
            canvas[i*224:(i+1)*224, j*224*2+224:(j+1)*224*2, :4] = recon_img
            img_idx += 1
    return (np.clip(canvas,0,1)*255).astype(np.uint8)


# ------------------------- train ---------------------------------------
def train(epoch,parser):
    model=parser.model
    model.train()
    running_loss = []
    running_img_loss = []
    running_lmk_loss = []
    running_creg_loss = []
    running_feat_loss = []
    running_gamma_loss = []
    running_reflect_loss = []
    loop = tqdm(enumerate(parser.train_dataloader), total=len(parser.train_dataloader))
    device=parser.device
    for i, data in loop:
        in_img, gt_img, lmk = data
        in_img = in_img.to(device); lmk = lmk.to(device)
        gt_img = gt_img.to(device)
        parser.optimizer.zero_grad()
        recon_params = model(in_img)
        loss, img_loss, lmk_loss, creg_loss, feat_loss, gamma_loss, reflect_loss,_ = parser.face_loss(recon_params, gt_img, lmk)
        loss.backward()
        parser.optimizer.step()
        running_loss.append(loss.item())
        running_img_loss.append(img_loss.item())
        running_lmk_loss.append(lmk_loss.item())
        running_creg_loss.append(creg_loss.item())
        running_feat_loss.append(feat_loss.item())
        running_gamma_loss.append(gamma_loss.item())
        running_reflect_loss.append(reflect_loss.item())
        loop.set_description("Loss: {:.6f}".format(np.mean(running_loss)))

        if i % parser.VERBOSE_STEP == 0 and i!=0:
            print ("Epoch: {:02}/{:02} Progress: {:05}/{:05} Loss: {:.6f} \
                   Img Loss: {:.6f} LMK Loss: {:.6f} Creg Loss {:.6f} \
                       Feat Loss {:.6f}  Gamma Loss {:.6f}  Reflect Loss {:.6f}".format(epoch+1, 
                                                                                                                    parser.NUM_EPOCH, 
                                                                                                                    i, 
                                                                                                                    len(parser.train_dataloader), 
                                                                                                                    np.mean(running_loss),
                                                                                                                    np.mean(running_img_loss),
                                                                                                                    np.mean(running_lmk_loss),
                                                                                                                    np.mean(running_creg_loss),
                                                                                                                    np.mean(running_feat_loss),
                                                                                                                    np.mean(running_gamma_loss),                                                                                                                 np.mean(running_reflect_loss)))
            running_loss = []
            running_img_loss = []
            running_lmk_loss = []
            running_creg_loss = []
            running_feat_loss = []
            running_gamma_loss = []
            running_reflect_loss = []
    return model

# ------------------------- eval ---------------------------------------
def eval(epoch,parser):
    model=parser.model
    model.eval()
    all_loss_list = []
    img_loss_list = []
    lmk_loss_list = []
    creg_loss_list = []
    feat_loss_list = []
    gamma_loss_list = []
    reflect_loss_list = []
    device=parser.device
    with torch.no_grad():
        for i, data in tqdm(enumerate(parser.val_dataloader), total=len(parser.val_dataloader)):
            in_img, gt_img, lmk = data
            in_img = in_img.to(device); lmk = lmk.to(device)
            gt_img = gt_img.to(device)
            recon_params = model(in_img)
            # import pdb; pdb.set_trace()
            all_loss, img_loss, lmk_loss, creg_loss, feat_loss, gamma_loss, reflect_loss, recon_img=parser.face_loss(recon_params, gt_img, lmk)
            all_loss_list.append(all_loss.item())
            img_loss_list.append(img_loss.item())
            lmk_loss_list.append(lmk_loss.item())
            creg_loss_list.append(creg_loss.item())
            feat_loss_list.append(feat_loss.item())
            gamma_loss_list.append(gamma_loss.item())
            reflect_loss_list.append(reflect_loss.item())
            if i == parser.VIS_BATCH_IDX:
                parser.visual_images.append(gt_img[0])
                parser.visual_faces.append(recon_img[0])
                visualize_image = visualize_batch(parser.VIS_COL_NUM, gt_img, recon_img)

    print ("-"*50, " Test Results ", "-"*50)
    _all_loss = np.mean(all_loss_list)
    _img_loss = np.mean(img_loss_list)
    _lmk_loss = np.mean(lmk_loss_list)
    _creg_loss = np.mean(creg_loss_list)
    _feat_loss = np.mean(feat_loss_list)
    _gamma_loss = np.mean(gamma_loss_list)
    _reflect_loss = np.mean(reflect_loss_list)
    print ("Epoch {:02}/{:02} all_loss: {:.6f} image loss: {:.6f} \
           landmark loss {:.6f} Creg loss {:.6f} \
               feat loss: {:.6f} gamma loss: {:.6f} reflect loss: {:.6f}".format(epoch+1, \
                   parser.NUM_EPOCH, _all_loss, _img_loss, _lmk_loss, _creg_loss,\
                       _feat_loss, _gamma_loss, _reflect_loss))
    print ("-"*116)
    return _all_loss, _img_loss, _lmk_loss, _creg_loss, _feat_loss, _gamma_loss, _reflect_loss, visualize_image

if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] ="1"
    opt = parse_opts()
    opt.visual_images=[]
    opt.visual_faces=[]
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # opt.device = torch.device('cpu')
    opt.renderer = sr.SoftRasterizer(image_size=224, sigma_val=1e-4, aggr_func_rgb='hard', fill_back=False)
    opt.transform = sr.LookAt(viewing_angle=30,perspective=True)
    opt.lighting = sr.Lighting(intensity_ambient=1.0,intensity_directionals=0)
    opt.transform.set_eyes_from_angles(opt.camera_distance, opt.elevation, opt.azimuth)
    opt.face_loss = BFMFaceLoss(opt)
    if not os.path.exists(opt.MODEL_SAVE_PATH):
        os.makedirs(opt.MODEL_SAVE_PATH)
    if not os.path.exists(opt.RESULT_SAVE_PATH):
        os.makedirs(opt.RESULT_SAVE_PATH)
    # -------------------------- Reproducibility ------------------------------
    torch.manual_seed(opt.seed)
    if not opt.device=='cuda:0':
        torch.backends.cudnn.benchmark = True
    if opt.accimage:
        torch.backends.torchvision.set_image_backend('accimage')
    
    # -------------------------- Dataset loading -----------------------------
    opt.train_set = CACDDataset(opt.Newdataset_path+"/CACD2000_train.hdf5", opt.train_transform, opt.inv_normalize)
    opt.val_set = CACDDataset(opt.Newdataset_path+"/CACD2000_val.hdf5", opt.val_transform, opt.inv_normalize)

    opt.train_dataloader = DataLoader(opt.train_set, batch_size=opt.BATCH_SIZE, shuffle=True)
    opt.val_dataloader = DataLoader(opt.val_set, batch_size=opt.BATCH_SIZE, shuffle=False)

    # -------------------------- Model loading ------------------------------
    opt.model = BaseModel(IF_PRETRAINED=False)
    opt.model.to(opt.device)
    opt.start_epoch = 0
    if opt.MODEL_LOAD_PATH is not None:
        sub_path=opt.MODEL_LOAD_PATH[opt.MODEL_LOAD_PATH.rfind('/')+7:]
        epoch_num=int(sub_path[:2])
        opt.start_epoch = epoch_num
        opt.model.load_state_dict(torch.load(opt.MODEL_LOAD_PATH)['model'])

    # -------------------------- Optimizer loading --------------------------
    opt.optimizer = optim.Adam(opt.model.parameters(), lr=opt.LR)
    opt.lr_schduler = optim.lr_scheduler.ReduceLROnPlateau(opt.optimizer, factor=0.2, patience=5)
    if opt.MODEL_LOAD_PATH is not None:
        opt.optimizer.load_state_dict(torch.load(opt.MODEL_LOAD_PATH)['optimizer'])

    # ------------------------- Loss training --------------------------------
    # all_loss, img_loss, lmk_loss, creg_loss, feat_loss, gamma_loss, reflect_loss, visualize_image = eval(opt.start_epoch, opt)
    # io.imsave(opt.RESULT_SAVE_PATH+"/training_progress.jpg", visualize_image)
    for epoch in range(opt.start_epoch,opt.NUM_EPOCH):
        opt.model = train(epoch, opt)
        all_loss, img_loss, lmk_loss, creg_loss, feat_loss, gamma_loss, reflect_loss, visualize_image = eval(epoch, opt)
        opt.lr_schduler.step(all_loss)
        io.imsave(opt.RESULT_SAVE_PATH+"/Epoch{:02}.png".\
              format(epoch+1), visualize_image)     
        model2save = {'model': opt.model.state_dict(),
                      'optimizer': opt.optimizer.state_dict()}
        torch.save(model2save, opt.MODEL_SAVE_PATH+"/epoch_{:02}_loss_{:.2f}_Img_loss_{:.2f}_LMK_loss{:.2f}_Creg_loss{:.2f}_Feat_loss{:.2f}_Gamma_loss{:.2f}_Ref_loss{:.2f}.pth".\
                format(epoch+1, all_loss, img_loss, lmk_loss, creg_loss,feat_loss, gamma_loss, reflect_loss))
        # ------------------------- Result visualization --------------------------------
        if epoch % 5==0:
            visualize_image=visualize_person(opt.VIS_COL_NUM, opt.visual_images, opt.visual_faces)
            io.imsave(opt.RESULT_SAVE_PATH+"/training_progress.jpg", visualize_image)
            img_list = sorted(glob.glob(os.path.join(opt.RESULT_SAVE_PATH+"/", "*.png")))
            cmd = ["ffmpeg", "-start_number", str(opt.video_start_epoch), "-r", str(opt.video_rate), '-i', os.path.join(opt.RESULT_SAVE_PATH+"/", "Epoch%02d.png"),str(epoch)+"_Face_Recon.mp4"]
            subprocess.run(cmd)
            


