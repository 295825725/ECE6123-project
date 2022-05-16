# ECE6123-project

BFM folder download url:https://drive.google.com/file/d/1WisaLaSeFqxbJ7l85CtH9cX4XkR2eAPq/view?usp=sharing

CACD2000 dataset url:https://drive.google.com/file/d/1hYIZadxcPG27Fo7mQln0Ey7uqw1DoBvM/view

data folder download url:https://drive.google.com/file/d/1QqjN10eX2yOF4e7nnt_0waVmOCktxH19/view?usp=sharing


trained model url:

put the model file in model_result_full folder



the model is running under python3.9, pytorch 1.11.0  with cuda11.3 cudnn8.0. 
It also requires:
* [softrenderer](https://github.com/ShichenLiu/SoftRas) (try to install this first)
* [Face Alignment](https://github.com/1adrianb/face-alignment)
* torchvision, tqdm, skimage, subprocess, numpy, h5py, scipy, tkinter, opencv, Pillow

you should change options.py to suit your device and path before runing the code

FaceLandmarkDetection.py will perform landmark detection, shuffle, and split operations on the input dataset into training, validation, and test files.

main.py will do the training, GUI.py and ReadAndCreate.py are the programs that put the model into practical use

Face_Recon.mp4 is a video of the training process, and gui_demo.mp4 is a demo video showing the results of GUI.py
