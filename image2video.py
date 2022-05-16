import glob, os
import subprocess
from options import parse_opts

opt = parse_opts()
img_list = sorted(glob.glob(os.path.join(opt.RESULT_SAVE_PATH+"/", "*.png")))
cmd = ["ffmpeg", "-start_number", str(20), "-r", str(opt.video_rate), '-i', os.path.join(opt.RESULT_SAVE_PATH+"/", "Epoch_%02d.png"),"Face_Recon.mp4"]
subprocess.run(cmd)
