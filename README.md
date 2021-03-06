# ECE6123-project
![result_example](https://user-images.githubusercontent.com/42291628/168714344-9aabfa2d-8022-457d-8f6e-157b7895749d.png)

BFM folder download url:https://drive.google.com/file/d/1WisaLaSeFqxbJ7l85CtH9cX4XkR2eAPq/view?usp=sharing

CACD2000 dataset url:https://drive.google.com/file/d/1hYIZadxcPG27Fo7mQln0Ey7uqw1DoBvM/view

data folder download url:https://drive.google.com/file/d/1QqjN10eX2yOF4e7nnt_0waVmOCktxH19/view?usp=sharing


trained model url:
https://drive.google.com/file/d/12vztGaSryc0l4Q2AuJO_HhYZTPVMjeHV/view?usp=sharing
https://drive.google.com/file/d/1AQ01MSx4CSy63REYXWQLIpLv3Jklamf5/view?usp=sharing

put the model file in model_result_full folder

![v2](https://user-images.githubusercontent.com/42291628/168712058-aba14068-3821-4c41-823d-1749de02e7fd.png)


the model is running under python3.9, pytorch 1.11.0  with cuda11.3 cudnn8.0. 
It also requires:
* [softrenderer](https://github.com/ShichenLiu/SoftRas) (try to install this first)
* [Face Alignment](https://github.com/1adrianb/face-alignment)
* torchvision, tqdm, skimage, subprocess, numpy, h5py, scipy, tkinter, opencv, Pillow

you should change options.py to suit your device and path before runing the code

FaceLandmarkDetection.py will perform landmark detection, shuffle, and split operations on the input dataset into training, validation, and test files.

main.py will do the training, GUI.py and ReadAndCreate.py are the programs that put the model into practical use

Face_Recon.mp4 is a video of the training process, and gui_demo.mp4 is a demo video showing the results of GUI.py
