# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:47:28 2022

@author: 29582
"""

import numpy as np
from torch.autograd import Variable
from PIL import ImageTk,Image
from model import BaseModel
import torch, cv2
from options import parse_opts
import soft_renderer as sr
from loss import BFMFaceLoss
import face_alignment
image_path="CACD2000/14_Aaron_Johnson_0001.jpg"
opt = parse_opts()
opt.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
opt.renderer = sr.SoftRasterizer(image_size=224, sigma_val=1e-4, aggr_func_rgb='hard', fill_back=False)
opt.transform = sr.LookAt(viewing_angle=30,perspective=True)
opt.lighting = sr.Lighting(intensity_ambient=1.0,intensity_directionals=0)
opt.transform.set_eyes_from_angles(opt.camera_distance, opt.elevation, opt.azimuth)
opt.face_loss = BFMFaceLoss(opt)
torch.manual_seed(opt.seed)
cv2image=cv2.imread(image_path)
cv2image= cv2.cvtColor(cv2image,cv2.COLOR_BGR2RGB)
if not opt.device=='cuda:0':
    torch.backends.cudnn.benchmark = True
if opt.accimage:
    torch.backends.torchvision.set_image_backend('accimage')
opt.model = BaseModel()
opt.model.to(opt.device)
opt.model.eval()
if opt.MODEL_LOAD_PATH is not None:
    opt.model.load_state_dict(torch.load(opt.MODEL_LOAD_PATH)['model'])
    
width  = cv2image.shape[0]
height = cv2image.shape[1]
_,_,faces = opt.fa.get_landmarks(cv2image,return_bboxes=True)
x_edge=70
y_edge=80
if faces is not None:
    faces=faces[0]
    x_min=max(0,int(faces[1])-x_edge)
    x_max=min(width,int(faces[3])+x_edge)
    y_min=max(0,int(faces[0])-y_edge)
    y_max=min(height,int(faces[2])+y_edge)
    recon_img=cv2image[x_min:x_max,y_min:y_max]
    image_tensor = opt.val_transform(recon_img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_image = Variable(image_tensor)
    input_image = input_image.to(opt.device)
    recon_params = opt.model(input_image)
    recon_img= opt.face_loss.reconst_img(recon_params).detach().cpu()
    recon_img = recon_img[0,:3,:,:].permute(1,2,0).numpy()
    cv2image= cv2.cvtColor(recon_img,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('output_3dface.jpg', cv2image)
    cv2.imshow("3D face",cv2image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("cannot detect any face")
    

    
