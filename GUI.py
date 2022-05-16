# -*- coding: utf-8 -*-
"""
Created on Sun May 15 10:13:19 2022

@author: 29582
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import tkinter as Tk   
import numpy as np
from torch.autograd import Variable
from PIL import ImageTk,Image
from model import BaseModel
import torch, cv2
from options import parse_opts
import soft_renderer as sr
from loss import BFMFaceLoss
import face_alignment


def fun_quit():
  global CONTINUE
  print('Good bye')
  CONTINUE = False
  
def fun_convert():
    global opt,cv2image,conv, recon_img
    conv= True
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
    else:
        recon_img=np.zeros(cv2image.shape)


conv = False
CONTINUE = True
cap = cv2.VideoCapture(0)
width  = cap.get(3)  
height = cap.get(4)
opt = parse_opts()
opt.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
opt.renderer = sr.SoftRasterizer(image_size=224, sigma_val=1e-4, aggr_func_rgb='hard', fill_back=False)
opt.transform = sr.LookAt(viewing_angle=30,perspective=True)
opt.lighting = sr.Lighting(intensity_ambient=1.0,intensity_directionals=0)
opt.transform.set_eyes_from_angles(opt.camera_distance, opt.elevation, opt.azimuth)
opt.face_loss = BFMFaceLoss(opt)
torch.manual_seed(opt.seed)
recon_img =None
if not opt.device=='cuda:0':
    torch.backends.cudnn.benchmark = True
if opt.accimage:
    torch.backends.torchvision.set_image_backend('accimage')
opt.model = BaseModel()
opt.model.to(opt.device)
opt.model.eval()
opt.start_epoch = 0
if opt.MODEL_LOAD_PATH is not None:
    sub_path=opt.MODEL_LOAD_PATH[opt.MODEL_LOAD_PATH.rfind('/')+7:]
    epoch_num=int(sub_path[:2])
    opt.start_epoch = epoch_num
    opt.model.load_state_dict(torch.load(opt.MODEL_LOAD_PATH)['model'])



# Define Tkinter root
root = Tk.Tk()
root.title('convert to 3D face')
   
# Define Tk variables
cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
img = Image.fromarray(cv2image)
label1 =Tk.Label(root)
label2 =Tk.Label(root)
# Convert image to PhotoImage

# Define widgets
# B_record = Tk.Button(root, text = 'convert', command = fun_convert, bg='green')
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit, bg='red')


# Place widgets
B_quit.grid(row=1,column=0, columnspan=5, sticky="news")
# B_record.grid(row=2,column=3)
label1.grid(row=0,column=0)
label2.grid(row=0,column=1)

while CONTINUE:
    root.update()
    cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = img)
    label1.imgtk = imgtk
    label1.configure(image=imgtk)
    fun_convert()
    img = recon_img* 255
    img = Image.fromarray(img.astype(np.uint8))
    imgtk = ImageTk.PhotoImage(image = img)
    label2.imgtk = imgtk
    label2.configure(image=imgtk)
        

cap.release()
root.destroy()
    
    