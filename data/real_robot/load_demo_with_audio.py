import os

from demonstration_withaudio import Demonstration
import matplotlib
matplotlib.use('TkAgg')
import time
from datetime import datetime
import wave
import logging
import cv2
import subprocess
import numpy as np
log_format = "%(asctime)s %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def demo2video(folder="/tmp/robot_demos/demo_2024-04-26T14-57-56-438644"):
    cur_t = datetime.now().strftime("%m-%d-%H:%M:%S")
    audio_path = os.path.join(folder + "/audio.wav")
    a = wave.open(audio_path, "rb")
    print(a.getparams())
    audio_len = a.getparams().nframes
    audio_data = a.readframes(nframes=audio_len)
    audio_data = np.frombuffer(audio_data, dtype=np.int16)
    audio_data = audio_data.reshape(audio_len, 1)
    print(audio_data.shape)


    image_folder = os.path.join(folder, "camera/220322060186/resample_rgb/")
    image_list = []
    ims = sorted(os.listdir(image_folder))
    for im in ims:
        image_list.append(cv2.imread(image_folder + im).copy())
    frequency = 10
    video_path = os.path.join(folder, "fix_cam.mp4")
    out = cv2.VideoWriter(video_path,
                          cv2.VideoWriter_fourcc(*"MJPG"), frequency,
                          (image_list[0].shape[1], image_list[0].shape[0]))  # careful of the size, should be W, H
    for frame in image_list:
        out.write(frame.astype('uint8').copy())
    out.release()
    mkv_save_path = os.path.join(folder, "output.mkv")
    cmd = f'ffmpeg -i {video_path} -i {audio_path} -c copy {mkv_save_path}'  # only mkv!!
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    folder = '/home/dia1rng/hackathon/6_6_rotate_cup 1/6_6_rotate_cup_org_data_samplingtime250'
    for traj in os.listdir(folder):
        if "demo" in traj:
            path = os.path.join(folder, traj)
            print(path)
            demo = Demonstration.load_from_folder(path)
            demo.resample_and_save(path, sampling_time=0.25, visualize=False)
            demo.get_info()
            demo2video(path)