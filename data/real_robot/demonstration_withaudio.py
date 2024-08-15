from multiprocessing import Queue, Process, Event, Lock
from typing import Optional, List, Dict
import logging
import random
import numpy as np
import time
import copy
import os
import datetime
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from klampt.model import trajectory
from klampt.math import so3

from tami_clap_candidate.rpc_interface.rpc_interface import RPCInterface, ArmState
from tami_clap_candidate.sensors.realsense import RealsenseRecorder, Preset
from tami_clap_candidate.pose_trajectory_processing.pose_trajectory_processor import Pose, RobotTrajectory
# import pyaudio
import wave
import subprocess
import shutil

logger = logging.getLogger(__name__)

class Demonstration:

    def __init__(self, **kwargs) -> None:
        # Config related variables
        self.source_ip = kwargs.get("source_robot_ip", None)
        self.target_ip = kwargs.get("target_robot_ip", None)
        self.sampling_time = kwargs.get("sampling_time", 0.1)
        self.resolution = kwargs.get("resolution", (640, 480))
        self.exposure = kwargs.get("exposure_time", -1)
        self.image_freq = kwargs.get("image_frequency", 30)
        logger.info(f"Using source robot IP: {self.source_ip}")
        logger.info(f"Using target robot IP: {self.target_ip}")
        logger.info(f"Using sampling time: {self.sampling_time}")

        # Class variables
        self.finish_event = Event()
        self.source_state_list = []
        self.target_state_list = []
        self.image_list = []
        self.audio_list = []
        self.source_state_q = Queue()
        self.target_state_q = Queue()
        self.image_q = Queue()
        self.audio_q = Queue()
        self.state_lock = Lock()
        self.image_lock = Lock()

        self.realsense_config = {
            "height": self.resolution[1],
            "width": self.resolution[0],
            "fps": self.image_freq,
            "record_depth": False,
            "depth_unit": 0.001,
            "preset": Preset.HighAccuracy,
            "memory_first": True,
            "exposure_time": self.exposure
        }

        # Resampling
        self.traj_resampled: Dict[str, RobotTrajectory] = {}
        self.traj_resampled_timestamps: List[float] = []

    def start_demo_recording(self, root_folder):
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        date_string = datetime.datetime.now().isoformat()
        folder_suffix = date_string.replace(":", "-").replace(".", "-")
        folder_name = os.path.join(root_folder, "demo_" + folder_suffix + "/")
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        self.folder_name = folder_name
        camera_folder_name = os.path.join(folder_name, "camera")
        self.realsense_config["record_folder"] = camera_folder_name

        processes = []
        # processes.append(Process(target=self.mic_process, daemon=True))
        processes.append(Process(target=self.realsense_process, args=(self.realsense_config,), daemon=True))
        if self.source_ip != None:
            processes.append(Process(target=self.source_robot_process, daemon=True))
        if self.target_ip != None:
            processes.append(Process(target=self.target_robot_process,  daemon=True))
        for p in processes:
            p.start()

        while True:
            cv2.imshow("Press q to exit", np.zeros((1, 1)))
            logger.info(f"Frame sizes (source, target, image): ({self.source_state_q.qsize(), self.target_state_q.qsize(), self.image_q.qsize()})")
            time.sleep(0.5)
            key = cv2.waitKey(1)
            if key == ord("q"):
                self.finish_event.set()
                break

        logger.info("Processes finished recording!")
        logger.info(f"Number of source state frames collected: {self.source_state_q.qsize()}")
        logger.info(f"Number of source state frames collected: {self.target_state_q.qsize()}")
        logger.info(f"Number of image frames collected: {self.image_q.qsize()}")

        time.sleep(1)

        logger.info(f"Converting from queues to lists...")
        self.source_state_list = []
        self.target_state_list = []
        self.image_list = []
        self.mic_frame_list = []
        while self.source_state_q.qsize() > 0:
            state_loc = self.source_state_q.get()
            self.source_state_list.append(state_loc)
        while self.target_state_q.qsize() > 0:
            state_loc = self.target_state_q.get()
            self.target_state_list.append(state_loc)
        while self.image_q.qsize() > 0:
            self.image_list.append(self.image_q.get())
        while self.audio_q.qsize() > 0:
            self.audio_list.append(self.audio_q.get())
        logger.info(f"... done")

    def report_record_frequency(self):
        plt.figure()
        start_time = None
        if len(self.source_state_list) > 0:
            time_stamps_state = np.array([step[1] for step in self.source_state_list])
            freq_state = 1.0 / (np.diff(time_stamps_state))
            start_time = time_stamps_state[0]
            time_stamps_state = time_stamps_state - start_time
            logger.info(f"Mean/std source state frequency: {np.mean(freq_state):.1f} / {np.std(freq_state):.1f} Hz")
            plt.subplot(211)
            plt.plot(time_stamps_state[1:], freq_state, 'b', label="source robot state")
            plt.subplot(212)
            plt.plot(time_stamps_state, 'b')
        else:
            logger.warning("Source state list is not yet filled, cannot plot frequency")
        if len(self.target_state_list) > 0:
            time_stamps_state = np.array([step[1] for step in self.target_state_list])
            freq_state = 1.0 / (np.diff(time_stamps_state))
            start_time = time_stamps_state[0]
            time_stamps_state = time_stamps_state - start_time
            logger.info(f"Mean/std target state frequency: {np.mean(freq_state):.1f} / {np.std(freq_state):.1f} Hz")
            plt.subplot(211)
            plt.plot(time_stamps_state[1:], freq_state, 'g', label="target robot state")
            plt.subplot(212)
            plt.plot(time_stamps_state, 'g')
        else:
            logger.warning("Target state list is not yet filled, cannot plot frequency")
        if len(self.image_list) > 0:
            time_stamps_image = np.array([step[1] for step in self.image_list])
            if start_time == None:
                start_time = time_stamps_image[0]
            time_stamps_image = time_stamps_image - start_time
            freq_image = 1.0 / (np.diff(time_stamps_image))
            logger.info(f"Mean/std image frequency: {np.mean(freq_image):.1f} / {np.std(freq_image):.1f} Hz")
            plt.subplot(211)
            plt.plot(time_stamps_image[1:], freq_image, 'r', label="image")
            plt.subplot(212)
            plt.plot(time_stamps_image, 'r')
        else:
            logger.warning("Image list is not yet filled, cannot plot frequency")
        plt.subplot(211)
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("frequency")
        plt.subplot(212)
        plt.xlabel("time steps")
        plt.ylabel("time")

    def realsense_process(self, realsense_config):
        logger.debug("Realsense process started")
        logger.debug("Starting realsense recorder ...")
        rs_recorder = RealsenseRecorder(
            **realsense_config,
        )
        rs_recorder._make_clean_folder(realsense_config["record_folder"])
        logger.debug("... realsense recorder intialized")
        while not self.finish_event.is_set():
            try:
                realsense_frames = rs_recorder.get_frame()
                self.image_q.put((copy.deepcopy(realsense_frames), time.time()))
            except (KeyboardInterrupt, SystemExit):
                break

        del rs_recorder

    def mic_process(self, mic_config=None):
        logger.debug("microphone process started")
        logger.debug("Starting microphone recorder ...")

        mic_idx = 6
        sr = 44100
        fps = 30
        CHUNK = int(sr / fps)
        # p = pyaudio.PyAudio()
        # audio_stream = p.open(format=pyaudio.paInt16,
        #                       channels=1,
        #                       rate=sr,
        #                       input=True,
        #                       input_device_index=mic_idx,  # Corrected variable name to microphone_index
        #                       frames_per_buffer=CHUNK)

        logger.debug("... microphone recorder intialized")
        # while not self.finish_event.is_set():
        #     try:
        #         audio_frames = audio_stream.read(CHUNK, exception_on_overflow=False)
        #         self.audio_q.put((copy.deepcopy(audio_frames), time.time()))
        #     except (KeyboardInterrupt, SystemExit):
        #         break
        #
        # del audio_stream

    def source_robot_process(self):
        logger.debug(f"Real robot process started with RPC @ {self.source_ip}")
        rpc = RPCInterface(self.source_ip)
        while not self.finish_event.is_set():
            arm_state = rpc.get_robot_state()
            self.source_state_q.put((arm_state, time.time()))

    def target_robot_process(self):
        logger.debug(f"Real robot process started with RPC @ {self.target_ip}")
        rpc = RPCInterface(self.target_ip)
        while not self.finish_event.is_set():
            arm_state = rpc.get_robot_state()
            self.target_state_q.put((arm_state, time.time()))


    def simulated_robot_process(self):
        logger.debug("Simulated robot process started")
        while not self.finish_event.is_set():
            try:
                delay = 1.0 / random.randint(25, 35)
                time.sleep(delay)
                arm_state = ArmState(np.random.randn(7).tolist(),
                                    np.random.randn(7).tolist(),
                                    Pose.origin(),
                                    np.random.randn(7).tolist(),
                                    np.random.randn(7).tolist(),
                                    np.random.randn(3).tolist(),
                                    np.random.randn(3).tolist(),
                                    [.1, .1])

                self.state_q.put((arm_state, time.time()))
            except (KeyboardInterrupt, SystemExit):
                break

    def get_info(self):
        logger.info(f"Q sizes (source, target, image): ({self.source_state_q.qsize(), self.target_state_q.qsize(), self.image_q.qsize()})")
        logger.info(f"List sizes (source, target, image): ({len(self.source_state_list), len(self.target_state_list), len(self.image_list)})")
        if hasattr(self, "source_traj_raw"):
            logger.info(f"Raw recording of source trajecotry length: {len(self.source_traj_raw.poses)}")
        if hasattr(self, "target_traj_raw"):
            logger.info(f"Raw recording of target trajecotry length: {len(self.target_traj_raw.poses)}")
        if len(self.traj_resampled) > 0:
            for key in self.traj_resampled.keys():
                logger.info(f"Resampled {key} trajectory length: {len(self.traj_resampled[key].poses)}")

    def save_raw_demos(self):
        def build_robot_traj(state_list, start_time):
            pose_list = []
            gripper_list = []
            time_stamp_list = []
            for pt in state_list:
                arm_state: ArmState = pt[0]
                time_stamp_list.append(pt[1] - start_time)
                pose_list.append(arm_state.pose)
                gripper_list.append(arm_state.gripper)
            traj = RobotTrajectory(pose_list, gripper_list, time_stamp_list)
            return traj

        plt.figure()
        if len(self.source_state_list) > 0:
            logger.debug("writing raw source robot trajectory to file ...")
            start_time = self.source_state_list[0][1]
            self.source_traj_raw = build_robot_traj(self.source_state_list, start_time)
            self.source_traj_raw.save_to_file(os.path.join(self.folder_name, "source_robot_trajectory.json"))

            pm, t = self.source_traj_raw.pose_matrix()
            plt.subplot(321)
            plt.title('raw source demo')
            plt.plot(t, pm[:, :3])
            plt.ylabel('pos')
            plt.subplot(323)
            plt.plot(t, pm[:, 3:7])
            plt.ylabel('quat')
            plt.subplot(325)
            plt.plot(t, pm[:, 7:])
            plt.ylabel('gripper')

        if len(self.target_state_list) > 0:
            logger.debug("writing raw target robot trajectory to file ...")
            self.target_traj_raw = build_robot_traj(self.target_state_list, start_time)
            self.target_traj_raw.save_to_file(os.path.join(self.folder_name, "target_robot_trajectory.json"))

            pm, t = self.target_traj_raw.pose_matrix()
            plt.subplot(322)
            plt.title("target raw demo")
            plt.plot(t, pm[:, :3])
            plt.subplot(324)
            plt.plot(t, pm[:, 3:7])
            plt.subplot(326)
            plt.plot(t, pm[:, 7:])

        if len(self.image_list) > 0:
            logger.debug("writing images to file")
            realsense_time_stamps = []
            rs_recorder = RealsenseRecorder(**self.realsense_config)
            pbar = tqdm(range(len(self.image_list)), desc="Writing images to file ...")
            for frame_count in pbar:
                rs_recorder._save_frame(self.image_list[frame_count][0], self.realsense_config['record_folder'], frame_count)
                realsense_time_stamps.append(self.image_list[frame_count][1])

            self.realsense_time_stamps = np.array(realsense_time_stamps) - start_time
            np.savetxt(os.path.join(self.folder_name, "image_time_stamps.txt"), self.realsense_time_stamps)

        if len(self.audio_list) > 0:
            logger.debug("writing audio to file")
            audio_time_stamps = []
            audio_frame_list = []
            for i in self.audio_list:
                audio_frame_list.append(i[0])
                audio_time_stamps.append(i[1])
            wf = wave.open(os.path.join(self.folder_name, 'audio.wav'), 'wb')
            wf.setnchannels(1)
            # print(audio.get_format_from_width(FORMAT).size)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b''.join(audio_frame_list))
            wf.close()
            self.mic_time_stamps = np.array(audio_time_stamps) - start_time
            np.savetxt(os.path.join(self.folder_name, 'audio_stamps.txt'), self.mic_time_stamps)

        logger.info(f'Files written to folder {self.folder_name}')

    @classmethod
    def load_from_folder(cls, folder_name):
        logger.info(f"loading demonstration from folder {folder_name}")
        if not os.path.exists(folder_name):
            raise ValueError(f"Folder {folder_name} not found!")
        source_path = os.path.join(folder_name, "source_robot_trajectory.json")
        target_path = os.path.join(folder_name, "target_robot_trajectory.json")
        image_time_stamp_path = os.path.join(folder_name, "image_time_stamps.txt")
        audio_time_stamp_path = os.path.join(folder_name, "audio_stamps.txt")

        c = cls()
        c.realsense_time_stamps = np.loadtxt(image_time_stamp_path)
        c.mic_time_stamps = np.loadtxt(audio_time_stamp_path)

        c.source_traj_raw = RobotTrajectory.from_path(source_path)
        c.target_traj_raw = RobotTrajectory.from_path(target_path)

        camera_folder = os.path.join(folder_name, "camera")
        serials = os.listdir(camera_folder)
        file_names = sorted(os.listdir(os.path.join(camera_folder, serials[0], "rgb")))
        c.image_list = []
        pbar = tqdm(range(len(file_names)), desc=f"Loading images  ...")
        for frame_count in pbar:
            cam_frame = {}
            for s in serials:
                cam_frame[s] = {"rgb": None, "depth": None, "intrinsic": None}
                serial_folder = os.path.join(camera_folder, s)
                rgb_folder = os.path.join(serial_folder, "rgb")
                rgb_frame = cv2.imread(os.path.join(rgb_folder, file_names[frame_count]))
                cam_frame[s]['rgb'] = rgb_frame
                with open(os.path.join(serial_folder, "intrinsics.json")) as f:
                    dat = json.load(f)
                cam_frame[s]['intrinsic'] = dat['intrinsic_matrix']
            c.image_list.append(cam_frame)
        return c

    def resample_and_save(self, folder_name: str, sampling_time: Optional[float]=None, visualize=False, save_audio=False):
        #generate smooth trajectories that can be interpolated and derived at arbitrary points in time
        if sampling_time == None:
            sampling_time = self.sampling_time
        logger.debug(f"Using sampling time {sampling_time}")

        traj_to_process: Dict[str, RobotTrajectory] = {}

        if hasattr(self, "source_traj_raw") and len(self.source_traj_raw.poses) > 0:
            traj_to_process["source"] = self.source_traj_raw
        if hasattr(self, "target_traj_raw") and len(self.target_traj_raw.poses) > 0:
            traj_to_process["target"] = self.target_traj_raw
        logger.debug(f"Available trajectory keys: {traj_to_process.keys()}")

        # Figure out min and max time_stamps
        image_start_time, image_end_time = self.realsense_time_stamps[0], self.realsense_time_stamps[-1]
        audio_start_time, audio_end_time = self.mic_time_stamps[0], self.mic_time_stamps[-1]
        start_time = max(image_start_time, audio_start_time)
        end_time = min(image_end_time, audio_end_time)
        for key in traj_to_process.keys():
            t_loc = traj_to_process[key].time_stamps
            if start_time < t_loc[0]:
                start_time = t_loc[0]
            if end_time > t_loc[-1]:
                end_time = t_loc[-1]
        logger.debug(f"Using time interval [{start_time}, {end_time}] with sampling time {sampling_time}")
        sampling_time_stamps = np.arange(start_time, end_time, sampling_time)
        self.traj_resampled_timestamps = sampling_time_stamps
        if save_audio:
            raw_audio = wave.open(os.path.join(folder_name, "audio.wav"), "rb")
            audio_len = raw_audio.getparams().nframes
            audio_data = raw_audio.readframes(nframes=audio_len)
            audio_len_real = audio_end_time - audio_start_time
            audio_start_resample = int((start_time - audio_start_time) / audio_len_real * audio_len)
            audio_end_resample = int((end_time - audio_start_time) / audio_len_real * audio_len)
            audio_data = np.frombuffer(audio_data, dtype=np.int16).reshape(-1, 1)
            audio_data = audio_data[audio_start_resample: audio_end_resample]
            wf = wave.open(os.path.join(folder_name, 'resampled_audio.wav'), 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio_data.tobytes())
            wf.close()

        re_t = np.expand_dims(sampling_time_stamps, axis=1)
        og_t = np.expand_dims(self.realsense_time_stamps, axis=0)
        og_t = np.tile(og_t, (re_t.shape[0], 1))
        diff = np.abs(og_t - re_t)
        pseudo_t = np.argmin(diff, axis=1, keepdims=False)

        for i in os.listdir(os.path.join(folder_name, "camera")):
            image_list = []
            image_folder = os.path.join(folder_name, "camera", i, "rgb")
            re_im_foler = os.path.join(folder_name, "camera", i, "resample_rgb")
            if not os.path.exists(re_im_foler):
                os.mkdir(re_im_foler)
            for idx, t in enumerate(pseudo_t):
                shutil.copyfile(os.path.join(image_folder, f"{t :06d}" + ".jpg"), os.path.join(re_im_foler, f"{idx :06d}" + ".jpg"))



        # resample the trajectories
        for traj_key in traj_to_process.keys():
            traj = traj_to_process[traj_key]
            logger.debug(f"resampling trajectory {traj_key}")

            pose_traj = trajectory.SE3Trajectory()
            gripper_traj = trajectory.Trajectory()

            t_raw = []
            # for state, action in zip(states.items(), actions.items()):
            for i in range(len(traj.poses)):
                pose = traj.poses[i].vec(quat_order="wxyz").tolist()
                time_stamp = traj.time_stamps[i]
                gripper = traj.grippers[i]

                t_raw.append(time_stamp)
                pose_traj.milestones.append(
                    so3.from_quaternion(pose[3:7]) + pose[0:3]
                )
                gripper_traj.milestones.append([gripper[0]])

            pose_traj.times = t_raw
            gripper_traj.times = t_raw

            # Make sure to add milestones at sampling time
            pose_traj.remesh(sampling_time_stamps)
            gripper_traj.remesh(sampling_time_stamps)


            new_pose_traj = []
            new_gripper_traj = []
            for t in sampling_time_stamps:
                new_pose = pose_traj.eval(t)
                pos = np.array(new_pose[1])
                quat = np.array(so3.quaternion(new_pose[0]))  #wxyz
                new_pose_traj.append(Pose.from_vec(np.concatenate((pos, quat))))
                new_gripper_traj.append(gripper_traj.eval(t)[0])

            self.traj_resampled[traj_key] = RobotTrajectory(new_pose_traj, new_gripper_traj, sampling_time_stamps)
            self.traj_resampled[traj_key].save_to_file(os.path.join(folder_name, f"resampled_{traj_key}_robot_trajectory.json"))
        np.savetxt(os.path.join(folder_name, "resampled_image_time_stamps.txt"), self.traj_resampled_timestamps)

        if visualize:
            for key, traj in self.traj_resampled.items():
                p, t = traj.pose_matrix()


                if key == "source":
                    pr, tr = self.source_traj_raw.pose_matrix()
                elif key == "target":
                    pr, tr = self.target_traj_raw.pose_matrix()

                plt.figure()
                plt.subplot(311)
                plt.plot(t, p[:, :3], 'o-')
                plt.plot(tr, pr[:, :3], 'x-')
                plt.ylabel('pos')
                plt.title(f"{key} robot trajectory")
                plt.subplot(312)
                plt.plot(t, p[:, 3:7], 'o-')
                plt.plot(tr, pr[:, 3:7], 'x-')
                plt.ylabel('quat')
                plt.subplot(313)
                plt.plot(t, p[:, 7:9], 'o-')
                plt.plot(tr, pr[:, 7:9], 'x-')
                plt.ylabel('gripper')
            plt.show()
