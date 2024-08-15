"""https://github.com/JunzheJosephZhu/see_hear_feel/tree/master/src/datasets"""
import copy
import json
import os
import random

from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
import wave
import os
import torch
import torchvision.transforms as T
from copy import deepcopy
from PIL import Image
from types import SimpleNamespace
from pose_trajectory_processor import PoseTrajectoryProcessor, RobotTrajectory
from quaternion import q_log_map, q_exp_map, exp_map_seq, log_map_seq, recover_pose_from_quat_real_delta, log_map, \
    exp_map
from omegaconf import OmegaConf, open_dict, ListConfig

from quaternion import smooth_traj


def get_pose_sequence(resampled_trajectory, lb_idx):
    return resampled_trajectory["pos_quat"][lb_idx]


class DummyDataset(Dataset):
    def __init__(self, traj_path, args, train=True):
        super().__init__()
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        self.traj_path = traj_path
        self.norm = args.norm_type if args.norm_type is not None else None
        self.norm_state = args.norm_state
        self.sampling_time = args.sampling_time / 1000
        self.smooth_factor = args.smooth_factor

        self.fix_cam_path = os.path.join(traj_path, "camera", "220322060186", "resample_rgb")
        self.gripper_cam_path = os.path.join(traj_path, "camera", "838212074210", "resample_rgb")
        self.pose_traj_processor = PoseTrajectoryProcessor()
        self.sr = 44100
        self.streams = [
            "cam_gripper_color",
            "cam_fixed_color",
        ]
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.len_obs = (self.num_stack - 1) * self.frameskip
        self.fps = int(1000 / args.sampling_time)
        self.resolution = (
                self.sr // self.fps
        )

        self.audio_len = self.num_stack * self.frameskip * self.resolution

        # augmentation parameters
        self.EPS = 1e-8

        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v

        self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))

        (self.resample_target_trajectory,
         self.smooth_resample_target_trajectory,
         ) = self.get_episode(traj_path, ablation=args.ablation,
                              sampling_time=args.sampling_time,
                              json_name="target_robot_trajectory.json",
                              smooth_factor=args.smooth_factor)

        if args.source:
            (self.resample_source_trajectory,
             self.smooth_resample_source_trajectory
             ) = self.get_episode(traj_path, ablation=args.ablation,
                                  sampling_time=args.sampling_time,
                                  json_name="source_robot_trajectory.json",
                                  smooth_factor=args.smooth_factor)
        else:
            (self.resample_source_trajectory,
             self.smooth_resample_source_trajectory
             ) = self.get_episode(traj_path, ablation=args.ablation,
                                  sampling_time=args.sampling_time,
                                  json_name="target_robot_trajectory.json",
                                  smooth_factor=args.smooth_factor)

        self.resample_target_trajectory["real_delta"], \
            self.resample_source_trajectory["real_delta"], \
            self.resample_target_trajectory["direct_vel"], = \
            self.compute_real_delta(self.resample_target_trajectory["pos_quat"],
                                    self.resample_source_trajectory["pos_quat"])

        self.smooth_resample_target_trajectory["real_delta"], \
            self.smooth_resample_source_trajectory["real_delta"], \
            self.smooth_resample_target_trajectory["direct_vel"], = \
            self.compute_real_delta(self.smooth_resample_target_trajectory["pos_quat"],
                                    self.smooth_resample_source_trajectory["pos_quat"])

        assert len(self.resample_source_trajectory["relative_real_time_stamps"]) == len(
            self.resample_target_trajectory["relative_real_time_stamps"])
        self.num_frames = len(self.resample_source_trajectory["relative_real_time_stamps"])
        print(f"{self.num_frames} of resampled frames from traj {traj_path}")

        self.modalities = args.ablation.split("_")
        self.no_crop = args.no_crop

        if self.train:
            self.transform_cam = [
                T.Resize((self.resized_height_v, self.resized_width_v), antialias=None),
                T.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02),
                # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
            self.transform_cam = T.Compose(self.transform_cam)

        else:
            self.transform_cam = T.Compose(
                [
                    T.Resize((self.resized_height_v, self.resized_width_v), antialias=None),
                    T.CenterCrop((self._crop_height_v, self._crop_width_v)),
                    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
            )
        self.len_lb = args.len_lb

        self.a_g, self.a_h = self.load_audio(traj_path, ablation=args.ablation)
        pass

    @staticmethod
    def get_episode(traj_path, ablation="", sampling_time=150, json_name=None, smooth_factor=None):
        """
        Return:
            folder for traj_path
            logs
            audio tracks
            number of frames in episode
        """

        def omit_nan(traj):
            last_t = 0.0
            for idx, (i, v) in enumerate(traj.items()):
                current_t = v["time"]
                assert current_t > last_t, "need to reorder trajectory dict"
                if np.isnan(v["pose"][3:]).any():
                    traj[i]["pose"][3:] = last_o
                #     v["pose"][3:][np.isnan(v["pose"][3:])] = robot_trajectory.items()[idx-1] + robot_trajectory.items()[idx+1]
                if np.isnan(v["pose"][3:]).any():
                    print("nan detected")
                last_t = current_t
                last_o = v["pose"][3:]
            return traj

        resample_traj_path = os.path.join(traj_path, f"resampled_{json_name}")
        assert os.path.exists(os.path.exists(resample_traj_path)), "resampled trajectory not exist!!"
        resampled_trajectory = RobotTrajectory.from_path(resample_traj_path)

        # get all resampled info
        resample_position_histroy = []
        resample_orientation_history = []
        for i in resampled_trajectory.poses:
            resample_orientation_history.append(
                [i.orientation.w, i.orientation.x, i.orientation.y, i.orientation.z])
            resample_position_histroy.append([i.position.x, i.position.y, i.position.z])

        # if np.isnan(np.array(resample_position_histroy)).any():
        #     print(f"nan detected in resampled pos histroy {json_name}")
        # if np.isnan(np.array(resample_orientation_history)).any():
        #     print(f"nan detected in resampled ori histroy {json_name}")
        smoothed_traj, global_delta, smoothed_global_delta = smooth_traj(
            np.concatenate([np.array(resample_position_histroy), np.array(resample_orientation_history)], axis=1),
            smooth_factor)

        resample_traj = {
            "pos_quat": np.concatenate([np.array(resample_position_histroy), np.array(resample_orientation_history)],
                                       axis=1),
            # "gripper": np.array([i[0] for i in resampled_trajectory.pose_trajectory.grippers]),
            "gripper": np.array(resampled_trajectory.grippers),
            "glb_pos_ori": global_delta,
            "relative_real_time_stamps": resampled_trajectory.time_stamps,
        }

        smooth_resample_traj = {
            "pos_quat": np.concatenate([smoothed_traj[:, :3], smoothed_traj[:, 3:]], axis=1),
            # "gripper": np.array([i[0] for i in resampled_trajectory.pose_trajectory.grippers]),
            "gripper": np.array(resampled_trajectory.grippers),
            "glb_pos_ori": smoothed_global_delta,
            "relative_real_time_stamps": resampled_trajectory.time_stamps,
        }

        return (
            resample_traj,
            smooth_resample_traj,
        )

    def compute_real_delta(self, target_pos_quat, source_pos_quat):
        target_real_delta = self.get_real_delta_sequence(target_pos_quat)
        source_real_delta = self.get_real_delta_sequence_direct(target_pos_quat, np.concatenate(
            [source_pos_quat[1:], source_pos_quat[-1:]], axis=0))
        direct_vel = self.get_real_delta_sequence_direct(target_pos_quat, source_pos_quat)
        return target_real_delta, source_real_delta, direct_vel

    def limit_norm(self, arr, min, max):
        gap = max - min
        arr = arr - min
        arr = arr / gap * 2
        arr = arr - 1
        return arr

    def gaussian_norm(self, arr, mean, std):
        arr = (arr - mean) / std
        return arr

    def normalize_audio(self, arr):
        statistic = copy.deepcopy(self.norm_state["resample"]["audio"])
        limit = max(np.abs(statistic["max"]), np.abs(statistic["min"]))
        return arr / limit

    def normalize(self, arr, state):
        if self.norm == "limit":
            return self.limit_norm(arr, state["min"], state["max"])
        elif self.norm == "gaussian":
            return self.gaussian_norm(arr, state["mean"], state["std"])
        else:
            return arr

    @staticmethod
    def load_audio(traj_path, ablation):
        modes = ablation.split("_")

        def load(file):
            fullpath = os.path.join(traj_path, file)
            if os.path.exists(fullpath):
                a = wave.open(fullpath, "rb")
                print(f"{traj_path} has audio with parameter:", a.getparams())
                audio_len = a.getparams().nframes
                audio_data = a.readframes(nframes=audio_len)
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
                # plt.plot(np.arange(audio_len), audio_data)
                # plt.show()
                return audio_data
            else:
                return None

        if "ag" in modes:
            audio_gripper = load("gripper_mic.wav")
            audio_gripper = [
                x for x in [audio_gripper] if x is not None
            ]
            audio_gripper = torch.as_tensor(np.stack(audio_gripper, 0))
        else:
            audio_gripper = None
        if "ah" in modes:
            audio_holebase = load("resampled_audio.wav")
            audio_holebase = [
                x for x in [audio_holebase] if x is not None
            ]
            audio_holebase = torch.as_tensor(np.stack(audio_holebase, 0))  # to [1, audio_len]
        else:
            audio_holebase = None

        return (
            audio_gripper,
            audio_holebase
        )

    @staticmethod
    def load_image(rgb_path, idx):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        rgb_path = os.path.join(rgb_path, idx)
        image = Image.open(rgb_path).convert('RGB')
        image = np.array(image)
        # cv2.imshow("show", image)
        # while True:
        #     key = cv2.waitKey(1)
        #     if key == ord("q"):
        #         break
        # image.show()
        image = torch.as_tensor(image).float().permute(2, 0, 1) / 255
        return image

    def clip_resample(self, audio, audio_start, audio_end):
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        if audio_start < 0:
            print("small!!!")
            left_pad = torch.zeros((audio.shape[0], -audio_start))
            audio_start = 0
        if audio_end >= audio.size(-1):
            print("large!!!")
            right_pad = torch.zeros((audio.shape[0], audio_end - audio.size(-1)))
            audio_end = audio.size(-1)
        audio_clip = torch.cat(
            [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
        )
        audio_clip = self.normalize_audio(audio_clip)
        audio_clip = torchaudio.functional.resample(audio_clip, 44100, 16000)
        return audio_clip

    @staticmethod
    def resize_image(image, size):
        assert len(image.size()) == 3  # [3, H, W]
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=size, mode="bilinear"
        ).squeeze(0)

    def __len__(self):
        return (self.num_frames - self.len_lb - self.len_obs - self.frameskip)

    @staticmethod
    def get_relative_delta_sequence(pos_quat: np.ndarray) -> (np.ndarray, np.ndarray):
        pos_delta_seq = pos_quat[:, :3]
        quat_seq = pos_quat[:, 3:]
        pos_base = pos_delta_seq[0:1, :]
        pos_delta_seq = pos_delta_seq - pos_base
        quat_seq = np.transpose(quat_seq, (1, 0))
        base = quat_seq[:, 0]
        ori_delta_seq = np.transpose(q_log_map(quat_seq, base), (1, 0))
        return np.concatenate((pos_delta_seq, ori_delta_seq), axis=1)

    @staticmethod
    def get_real_delta_sequence(pos_quat: np.ndarray) -> (np.ndarray, np.ndarray):
        # if np.isnan(quaternion_seq).any():
        #     print("nan detected before")
        pos_base = pos_quat[:, :3]
        o_base = pos_quat[:, 3:]
        pos_new = np.concatenate((pos_base[1:, :], pos_base[-1:, :]), axis=0)
        pos_delta_seq = pos_new - pos_base
        o_delta_seq = np.zeros([o_base.shape[0], 3])
        o_base = np.transpose(o_base, (1, 0))
        o_new = np.concatenate((o_base[:, 1:].copy(), o_base[:, -1:].copy()), axis=1)
        for i in range(o_delta_seq.shape[0]):
            o_delta_seq[i, :] = q_log_map(o_new[:, i], o_base[:, i])
        return np.concatenate((pos_delta_seq, o_delta_seq), axis=1)

    @staticmethod
    def get_real_delta_sequence_direct(target_pos_quat, source_pos_quat) -> (np.ndarray, np.ndarray):
        # if np.isnan(quaternion_seq).any():
        #     print("nan detected before")
        direct_real_delta = np.zeros([target_pos_quat.shape[0], 6])
        for i in range(len(target_pos_quat)):
            direct_real_delta[i] = log_map(source_pos_quat[i], target_pos_quat[i])
        return direct_real_delta

    def get_traj_info(self,
                      source_trajectory,
                      target_trajectory,
                      seq_idx,
                      delta_idx,
                      norm_state):
        output = {}
        output["source_pos_quat"] = source_trajectory["pos_quat"][seq_idx]
        output["source_glb_pos_ori"] = source_trajectory["glb_pos_ori"][seq_idx]
        output["source_real_delta"] = source_trajectory["real_delta"][delta_idx]

        output["target_pos_quat"] = target_trajectory["pos_quat"][seq_idx]
        output["target_glb_pos_ori"] = target_trajectory["glb_pos_ori"][seq_idx]
        output["target_real_delta"] = target_trajectory["real_delta"][delta_idx]

        output["gripper"] = target_trajectory["gripper"][seq_idx]
        output["direct_vel"] = target_trajectory["direct_vel"][delta_idx]

        output = {key: torch.tensor(self.normalize(value, state=norm_state[key]) if "quat" not in key else value) for
                  key, value in output.items()}
        output = {key: {"obs": value[:-self.len_lb], "action": value[-self.len_lb:]} for key, value in output.items()}
        return output

    def __getitem__(self, idx):
        # print("idx", idx)
        idx = idx + self.len_obs + self.frameskip
        start = idx - self.len_obs
        if start < 0:
            print("image small!!!!")
        # compute which frames to use
        frame_idx = np.arange(start, idx + 1, self.frameskip)
        frame_idx[frame_idx < 0] = 0

        lb_end = idx + self.len_lb
        pose_lb_idx = np.concatenate([frame_idx, np.arange(idx + 1, lb_end + 1)])
        pose_lb_idx[pose_lb_idx >= self.num_frames] = -1

        pose_lb_idx[pose_lb_idx < 0] = 0
        pose_idx = copy.deepcopy(pose_lb_idx)

        delta_lb_idx = np.concatenate([frame_idx - 1, np.arange(idx, lb_end)])
        delta_lb_idx[delta_lb_idx >= self.num_frames] = -1

        delta_lb_idx[delta_lb_idx < 0] = 0
        delta_idx = copy.deepcopy(delta_lb_idx)

        traj_info = self.get_traj_info(
            self.resample_source_trajectory,
            self.resample_target_trajectory,
            pose_idx,
            delta_idx,
            self.norm_state["resample"])

        smooth_traj_info = self.get_traj_info(
            self.smooth_resample_source_trajectory,
            self.smooth_resample_target_trajectory,
            pose_idx,
            delta_idx,
            self.norm_state["smooth"])

        # 2i_images
        # to speed up data loading, do not load img if not using
        cam_gripper_framestack = 0
        cam_fixed_framestack = 0

        # process different streams of data
        if "vg" in self.modalities:
            cam_gripper_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.gripper_cam_path, f"{timestep :06d}" + ".jpg")
                    )
                    for timestep in frame_idx
                ],
                dim=0,
            )
        if "vf" in self.modalities:
            cam_fixed_framestack = torch.stack(
                [
                    self.transform_cam(
                        self.load_image(self.fix_cam_path, f"{timestep :06d}" + ".jpg")
                    )
                    for timestep in frame_idx
                ],
                dim=0,
            )

        # random cropping
        if self.train:  # get random crop params (previously resize and color jitter)
            img = self.transform_cam(
                self.load_image(self.gripper_cam_path, f"{idx:06d}" + ".jpg")
            )
            if not self.no_crop:
                i_v, j_v, h_v, w_v = T.RandomCrop.get_params(
                    img, output_size=(self._crop_height_v, self._crop_width_v)
                )
            else:
                i_v, h_v = (
                                   self.resized_height_v - self._crop_height_v
                           ) // 2, self._crop_height_v
                j_v, w_v = (
                                   self.resized_width_v - self._crop_width_v
                           ) // 2, self._crop_width_v

            if "vg" in self.modalities:
                cam_gripper_framestack = cam_gripper_framestack[
                                         ..., i_v: i_v + h_v, j_v: j_v + w_v
                                         ]
            if "vf" in self.modalities:
                cam_fixed_framestack = cam_fixed_framestack[
                                       ..., i_v: i_v + h_v, j_v: j_v + w_v
                                       ]

        # load audio
        # if idx == 88:
        #     print(idx)
        audio_end = (idx + 1) * self.resolution
        audio_start = audio_end - self.audio_len  # why self.sr // 2, and start + sr
        # if self.a_g is not None:
        #     audio_clip_g = self.clip_resample(
        #         self.a_g, audio_start, audio_end
        #     ).float()
        # else:
        #     audio_clip_g = 0
        # if self.a_h is not None:
        #     audio_clip_h = self.clip_resample(
        #         self.a_h, audio_start, audio_end
        #     ).float()
        # else:
        #     audio_clip_h = 0

        return {
            "traj": traj_info,
            "smooth_traj": smooth_traj_info,
            "observation": {"v_fix": cam_fixed_framestack,
                            "v_gripper": cam_gripper_framestack},
            # "a_holebase": audio_clip_h,
            # "a_gripper": audio_clip_g},
            "start": start,
            "current": idx,
            "end": lb_end,
            "traj_idx": self.traj_path,
        }


class Normalizer(Dataset):
    def __init__(self, traj_folder_path, args, norm_state=None, **kwargs):
        super().__init__()
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        self.traj_folder_path = traj_folder_path
        self.args = args
        self.pose_traj_processor = PoseTrajectoryProcessor()

        if norm_state is None:
            state, smooth_state = self.get_all_traj_state(args)
            norm_state = self.get_state(state)
            smooth_norm_state = self.get_state(smooth_state)
            self.norm_state = {"resample": norm_state, "smooth": smooth_norm_state}
        else:
            self.norm_state = norm_state

    def get_all_traj_state(self, args):
        target_real_delta = []
        target_glb_pos_ori = []
        smooth_target_real_delta = []
        smooth_target_glb_pos_ori = []

        source_real_delta = []
        source_glb_pos_ori = []
        smooth_source_real_delta = []
        smooth_source_glb_pos_ori = []

        gripper = []
        direct_vel = []
        smooth_direct_vel = []

        # audio = []

        for traj in self.traj_folder_path:
            (
                resample_target_trajectory,
                smooth_resample_target_trajectory,
            ) = DummyDataset.get_episode(traj, ablation=args.ablation,
                                         sampling_time=args.sampling_time,
                                         json_name="target_robot_trajectory.json",
                                         smooth_factor=self.args.smooth_factor)

            if args.source:
                (
                    resample_source_trajectory,
                    smooth_resample_source_trajectory
                ) = DummyDataset.get_episode(traj, ablation=args.ablation,
                                             sampling_time=args.sampling_time,
                                             json_name="source_robot_trajectory.json",
                                             smooth_factor=self.args.smooth_factor)
            else:
                (
                    resample_source_trajectory,
                    smooth_resample_source_trajectory
                ) = DummyDataset.get_episode(traj, ablation=args.ablation,
                                             sampling_time=args.sampling_time,
                                             json_name="target_robot_trajectory.json",
                                             smooth_factor=self.args.smooth_factor)

            resample_target_trajectory["real_delta"], \
                resample_source_trajectory["real_delta"], \
                resample_target_trajectory["direct_vel"], = \
                self.compute_real_delta(resample_target_trajectory["pos_quat"],
                                        resample_source_trajectory["pos_quat"])

            smooth_resample_target_trajectory["real_delta"], \
                smooth_resample_source_trajectory["real_delta"], \
                smooth_resample_target_trajectory["direct_vel"], = \
                self.compute_real_delta(smooth_resample_target_trajectory["pos_quat"],
                                        smooth_resample_source_trajectory["pos_quat"])

            # a_g, a_h = DummyDataset.load_audio(traj, args.ablation)
            #
            # audio.append(a_h[0].detach().cpu().numpy())

            source_glb_pos_ori.append(resample_source_trajectory["glb_pos_ori"])
            source_real_delta.append(resample_source_trajectory["real_delta"])

            target_glb_pos_ori.append(resample_target_trajectory["glb_pos_ori"])
            target_real_delta.append(resample_target_trajectory["real_delta"])

            gripper.append(resample_target_trajectory["gripper"])
            direct_vel.append(resample_target_trajectory["direct_vel"])

            smooth_source_glb_pos_ori.append(smooth_resample_source_trajectory["glb_pos_ori"])
            smooth_source_real_delta.append(smooth_resample_source_trajectory["real_delta"])

            smooth_target_glb_pos_ori.append(smooth_resample_target_trajectory["glb_pos_ori"])
            smooth_target_real_delta.append(smooth_resample_target_trajectory["real_delta"])

            smooth_direct_vel.append(smooth_resample_target_trajectory["direct_vel"])

        state = {}

        # state["audio"] = np.concatenate(audio, axis=0)
        state["target_real_delta"] = np.concatenate(target_real_delta, axis=0)
        state["target_glb_pos_ori"] = np.concatenate(target_glb_pos_ori, axis=0)
        state["source_real_delta"] = np.concatenate(source_real_delta, axis=0)
        state["source_glb_pos_ori"] = np.concatenate(source_glb_pos_ori, axis=0)
        state["gripper"] = np.concatenate(gripper, axis=0)
        state["direct_vel"] = np.concatenate(direct_vel, axis=0)

        smooth_state = {}
        smooth_state["target_real_delta"] = np.concatenate(smooth_target_real_delta, axis=0)
        smooth_state["target_glb_pos_ori"] = np.concatenate(smooth_target_glb_pos_ori, axis=0)
        smooth_state["source_real_delta"] = np.concatenate(smooth_source_real_delta, axis=0)
        smooth_state["source_glb_pos_ori"] = np.concatenate(smooth_source_glb_pos_ori, axis=0)
        smooth_state["gripper"] = np.concatenate(gripper, axis=0)
        smooth_state["direct_vel"] = np.concatenate(smooth_direct_vel, axis=0)

        return state, smooth_state

    def save_json(self, save_json_path=None):
        def to_json(non_json):
            if isinstance(non_json, np.ndarray):
                return non_json.tolist()
            elif isinstance(non_json, PoseTrajectoryProcessor):
                return non_json.__dict__
            elif isinstance(non_json, SimpleNamespace):
                return non_json.__dict__
            elif isinstance(non_json, ListConfig):
                return list(non_json)
            elif isinstance(non_json, np.int16):
                return non_json.item()
            else:
                return non_json

        state_dict = json.dumps(self.__dict__, default=to_json, indent=4)

        if save_json_path is not None:
            save_json_path = os.path.join(save_json_path, "normalizer_config.json")
        else:
            save_json_path = os.path.abspath(os.path.join(self.traj_folder_path[0], "..", "normalizer_config.json"))
        with open(save_json_path, 'w') as json_file:
            json_file.write(state_dict)
        return

    @classmethod
    def from_json(cls, json_str):

        def dict_values_to_np_arrays(input_dict):
            """
            Transforms every value inside the dictionary to NumPy arrays.

            Parameters:
            - input_dict (dict): Input dictionary

            Returns:
            - dict: Dictionary with values transformed to NumPy arrays
            """
            output_dict = {}
            for key, value in input_dict.items():
                if isinstance(value, dict):
                    output_dict[key] = dict_values_to_np_arrays(value)  # Recursively handle nested dictionaries
                elif isinstance(value, (list, tuple)):
                    output_dict[key] = np.array(value)  # Convert lists/tuples to NumPy arrays
                elif isinstance(value, np.ndarray):
                    output_dict[key] = value  # No need to convert if already a NumPy array
                else:
                    output_dict[key] = np.array([value])  # Convert other types to 1D NumPy arrays
            return output_dict

        json_str = os.path.join(json_str, "normalizer_config.json")
        with open(json_str, 'r') as j:
            data = json.loads(j.read())

        norm_state = dict_values_to_np_arrays(data["norm_state"])

        data["norm_state"] = norm_state

        # print(norm_state)

        data["args"] = SimpleNamespace(**data["args"])

        return cls(**data)

    def compute_real_delta(self, target_pos_quat, source_pos_quat):
        target_real_delta = DummyDataset.get_real_delta_sequence(target_pos_quat)
        source_real_delta = DummyDataset.get_real_delta_sequence_direct(target_pos_quat, np.concatenate(
            [source_pos_quat[1:], source_pos_quat[-1:]], axis=0))
        direct_vel = DummyDataset.get_real_delta_sequence_direct(target_pos_quat, source_pos_quat)
        return target_real_delta, source_real_delta, direct_vel

    @staticmethod
    def compute_max_min_mean_std(arr):
        return {"max": np.max(arr, axis=0),
                "min": np.min(arr, axis=0),
                "mean": np.mean(arr, axis=0),
                "std": np.std(arr, axis=0)}

    def get_state(self, state):
        return {key: self.compute_max_min_mean_std(value) for key, value in state.items()}

    def denormalize(self, x, var_name):
        print('denormalize ', var_name)
        statistic = copy.deepcopy(self.norm_state[self.args.catg][var_name])
        if isinstance(x, torch.Tensor):
            for k, v in statistic.items():
                statistic[k] = torch.from_numpy(v).to(x.device)
        if self.args.norm_type == "limit":
            return (x + 1) / 2 * (statistic["max"] - statistic["min"]) + statistic["min"]
        elif self.args.norm_type == "gaussian":
            return x * statistic["std"] + statistic["mean"]
        else:
            raise TypeError("norm type unrecognized")

    def denormalize_audio(self, audio):
        statistic = copy.deepcopy(self.norm_state["resample"]["audio"])
        limit = max(np.abs(statistic["max"]), np.abs(statistic["min"]))
        return audio * limit


def get_loaders(batch_size: int, args, data_folder: str, drop_last: bool, save_json=None, debug=False,
                val_batch_size=64, **kwargs):
    """

    Args:
        batch_size: batch size
        args: arguments for dataloader
        data_folder: absolute path of directory "data"
        drop_last: whether drop_last for train dataloader
        **kwargs: other arguments

    Returns: training loader and validation loader

    """
    args = SimpleNamespace(**args) if not isinstance(args, SimpleNamespace) else args
    args.save_json = save_json

    trajs = [os.path.join(data_folder, traj) for traj in sorted(os.listdir(data_folder)) if "demo" in traj]
    num_train = int(len(trajs) * 0.95)

    train_trajs_paths = trajs[:num_train]
    val_trajs_paths = trajs[num_train:]
    if debug:
        start_id = 5
        train_trajs_paths = train_trajs_paths[0:1]
        val_trajs_paths = trajs[start_id:start_id + 1]

    normalizer = Normalizer(train_trajs_paths, args)
    normalizer.save_json(save_json)
    args.norm_state = normalizer.norm_state
    print("normalization state:", args.norm_state)

    train_set = torch.utils.data.ConcatDataset(
        [
            DummyDataset(traj, args, )
            for i, traj in enumerate(train_trajs_paths)
        ]
    )

    train_inference_set = torch.utils.data.ConcatDataset(
        [
            DummyDataset(traj, args, False)
            for i, traj in enumerate(train_trajs_paths)
        ]
    )

    val_set = torch.utils.data.ConcatDataset(
        [
            DummyDataset(traj, args, False)
            for i, traj in enumerate(val_trajs_paths)
        ]
    )

    train_loader = DataLoader(train_set, batch_size, num_workers=8, shuffle=True, drop_last=drop_last, )
    print(f"number of training trajectories: {len(train_trajs_paths)} \n train loader length: {len(train_loader)} \n "
          f"batch size: {batch_size} \n in total {batch_size * len(train_loader)} training samples", )
    train_inference_loader = DataLoader(train_inference_set, 1, num_workers=8, shuffle=False, drop_last=False, )
    if not debug:
        val_loader = DataLoader(val_set, val_batch_size, num_workers=8, shuffle=False, drop_last=False, )
    else:
        val_loader = DataLoader(val_set, 1, num_workers=8, shuffle=False, drop_last=False, )
    print(
        f"number of validation trajectories: {len(val_trajs_paths)} \n validation loader length: {len(val_loader)} \n", )
    return train_loader, val_loader, train_inference_loader


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import time
    import cv2


    def plot_2arr(l, name):

        for tensor, n in zip(l, name):
            print(f"{name} max = {np.max(tensor.detach().cpu().numpy(), axis=0)}")
            print(f"{name} min = {np.min(tensor.detach().cpu().numpy(), axis=0)}")
            print(f"{name} mean = {np.mean(tensor.detach().cpu().numpy(), axis=0)}")
            print(f"{name} std = {np.std(tensor.detach().cpu().numpy(), axis=0)}")
        arr = np.stack(l, axis=-1).transpose([1, 2, 0])
        num_channels = arr.shape[0]
        len = arr.shape[2]
        fig, axs = plt.subplots(num_channels, 1, figsize=(len, num_channels))

        t = np.arange(len)

        for idx in range(num_channels):
            for i in range(arr.shape[1]):
                axs[idx].plot(t, arr[idx][i], '-', label=name[i])
        plt.legend()
        plt.show()


    data_folder_path = '/home/dia1rng/hackathon/4_29_pouring'
    args = SimpleNamespace()

    args.ablation = 'vg_vf'
    args.num_stack = 2
    args.frameskip = 2
    args.no_crop = True
    args.crop_percent = 0.0
    args.resized_height_v = 240
    args.resized_width_v = 320
    args.len_lb = 10
    args.sampling_time = 100

    args.source = True

    args.smooth_factor = (1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5)

    args.catg = "resample"
    args.norm_type = "limit"

    train_loader, val_loader, train_inference_loader, = get_loaders(batch_size=1, args=args,
                                                                    data_folder=data_folder_path,
                                                                    drop_last=False, debug=True, save_json=None)
    norm_state = train_loader.dataset.datasets[0].norm_state

    normalizer = Normalizer.from_json(data_folder_path)
    ######## show images ###################################################################
    # print(len(val_loader))
    #
    for idx, batch in enumerate(val_loader):
        # if idx >= 100:
        #     break
        print(f"{idx} \n")
        obs = batch["observation"]

        image_f = obs["v_fix"][0][-1].permute(1, 2, 0).numpy()
        image_g = obs["v_gripper"][0][-1].permute(1, 2, 0).numpy()
        image = np.concatenate([image_f, image_g], axis=0)
        plt.imshow(image)
        plt.draw()
        plt.pause(0.2)
        plt.clf()
        # cv2.imshow("asdf", image)
        # time.sleep(0.1)
        # key = cv2.waitKey(1)
        # if key == ord("q"):
        #     break
    plt.close()
    ######## show images ###################################################################

    ######## show images and audio #################################################################
    print(len(val_loader))
    # audio_list = []
    # for idx, batch in enumerate(train_inference_loader):
    #     # if idx >= 100:
    #     #     break
    #     print(f"{idx} \n")
    #     obs = batch["observation"]
    #
    #     image_f = obs["v_fix"][0][-1].permute(1, 2, 0).numpy()
    #     image_g = obs["v_gripper"][0][-1].permute(1, 2, 0).numpy()
    #     # audio = obs["a_holebase"][0][-1][-1600:].numpy()
    #     # audio = normalizer.denormalize_audio(audio=audio)
    #     # stream = audio.open(format=FORMAT, channels=1, rate=16000, output=True, frames_per_buffer=1600,
    #     #                     output_device_index=0)
    #     # for i in range(len(frames)):
    #     #     stream.write(frames[i])
    #     #
    #     # stream.stop_stream()
    #     # stream.close()
    #
    #     audio_list.append(audio)
    #
    #     image = np.concatenate([image_f, image_g], axis=0)
    #     cv2.imshow("asdf", image)
    #     time.sleep(0.1)
    #     key = cv2.waitKey(1)
    #     if key == ord("q"):
    #         break
    # audio_list = np.concatenate(audio_list, axis=0)
    # plt.plot(np.arange(audio_list.shape[0]), audio_list)
    # plt.show()
    ####### show image and audio ########################################################################

    #######check pose and vel######################################################################3
    all_step_source_pose = []
    all_step_target_pose = []
    all_step_gripper = []
    all_step_source_pos_ori = []
    all_step_target_pos_ori = []

    all_step_source_real_delta = []
    all_step_target_real_delta = []
    all_step_direct_vel = []

    for idx, batch in enumerate(val_loader):
        # if idx >= 100:
        #     break
        print(f"{idx} \n")
        obs = batch["observation"]

        # for image_g in obs[1][0]:
        #     cv2.imshow("asdf", image_g.permute(1, 2, 0).numpy())
        # key = cv2.waitKey(1)
        # if key == ord("q"):
        #     break
        all_step_gripper.append(batch["traj"]["gripper"]["obs"][0, -1:].unsqueeze(-1))

        all_step_source_pose.append(batch["traj"]["source_pos_quat"]["obs"][0, -1:, :])
        all_step_target_pose.append(batch["traj"]["target_pos_quat"]["obs"][0, -1:, :])

        all_step_source_real_delta.append(batch["traj"]["source_real_delta"]["obs"][0, -1:, :])
        all_step_target_real_delta.append(batch["traj"]["target_real_delta"]["obs"][0, -1:, :])
        all_step_direct_vel.append(batch["traj"]["direct_vel"]["obs"][0, -1:, :])

        all_step_source_pos_ori.append(batch["traj"]["source_glb_pos_ori"]["obs"][0, -1:, :])
        all_step_target_pos_ori.append(batch["traj"]["target_glb_pos_ori"]["obs"][0, -1:, :])

    all_step_gripper = torch.cat(all_step_gripper, dim=0)

    all_step_source_pose = torch.concatenate(all_step_source_pose, dim=0)
    all_step_target_pose = torch.concatenate(all_step_target_pose, dim=0)
    plot_2arr([torch.cat([all_step_target_pose, all_step_gripper], dim=-1),
               torch.cat([all_step_source_pose, all_step_gripper], dim=-1)],
              ["target", "source"])

    all_step_target_pos_ori = torch.cat(all_step_target_pos_ori, dim=0)
    # all_step_target_pos_ori = normalizer.denormalize(all_step_target_pos_ori, "target_glb_pos_ori")
    all_step_source_pos_ori = torch.cat(all_step_source_pos_ori, dim=0)
    # all_step_source_pos_ori = normalizer.denormalize(all_step_source_pos_ori, "source_glb_pos_ori")
    plot_2arr([torch.cat([all_step_source_pos_ori, all_step_gripper], dim=-1)[:, 0:-1:2],
               torch.cat([all_step_target_pos_ori, all_step_gripper], dim=-1)[:, 0:-1:2]],
              ["source", "target"])

    all_step_source_real_delta = torch.cat(all_step_source_real_delta, dim=0)
    all_step_target_real_delta = torch.cat(all_step_target_real_delta, dim=0)
    all_step_direct_vel = torch.cat(all_step_direct_vel, dim=0)
    plot_2arr([all_step_direct_vel, all_step_target_real_delta, all_step_source_real_delta],
              ["direct", "target", "source"])
    #######check pos and vel######################################################################3

    #######check pose recover######################################################################3
    # all_step_pose = []
    # all_recover_real_delta_pose = []
    # all_recover_direct_vel_pose = []
    # for idx, batch in enumerate(val_loader):
    #     if idx % args.len_lb != 0:
    #         continue
    #     # if idx >= 100:
    #     #     break
    #     print(f"{idx} \n")
    #     obs = batch["observation"]
    #
    #     # for image_g in obs[1][0]:
    #     #     cv2.imshow("asdf", image_g.permute(1, 2, 0).numpy())
    #     # key = cv2.waitKey(1)
    #     # if key == ord("q"):
    #     #     break
    #
    #     all_step_pose.append(batch["traj"]["target_pos_quat"]["action"][0, :, :])
    #
    #     recover_real_delta_pose = recover_pose_from_quat_real_delta((batch["traj"]["target_real_delta"]["action"][0, :, :].detach().cpu().numpy() + 1) / 2 * (norm_state["resample"]["target_real_delta"]["max"] - norm_state["resample"]["target_real_delta"]["min"]) + norm_state["resample"]["target_real_delta"]["min"],
    #                                batch["traj"]["target_pos_quat"]["obs"][0, -1, :].detach().cpu().numpy(), )
    #
    #     recover_direct_vel_pose = recover_pose_from_quat_real_delta((batch["traj"]["direct_vel"]["action"][0, :, :].detach().cpu().numpy() + 1) / 2 * (norm_state["resample"]["direct_vel"]["max"] - norm_state["resample"]["direct_vel"]["min"]) + norm_state["resample"]["direct_vel"]["min"],
    #                                batch["traj"]["target_pos_quat"]["obs"][0, -1, :].detach().cpu().numpy(), )
    #     all_recover_real_delta_pose.append(torch.from_numpy(recover_real_delta_pose))
    #     all_recover_direct_vel_pose.append(torch.from_numpy(recover_direct_vel_pose))
    #
    # all_step_pose = torch.concatenate(all_step_pose, dim=0)
    # all_recover_real_delta_pose = torch.concatenate(all_recover_real_delta_pose, dim=0)
    # all_recover_direct_vel_pose = torch.cat(all_recover_direct_vel_pose, dim=0)
    #
    # plot_2arr([all_step_pose, all_recover_real_delta_pose, all_recover_direct_vel_pose], ["og", "recover_target_real_delta", "direct_vel"])
    #######check pose recover######################################################################3

    #######check delta and velocity######################################################################3
    # all_step_source_pose = []
    # all_step_target_pose = []
    # all_step_gripper = []
    # all_step_source_pos_ori = []
    # all_step_target_pos_ori = []
    #
    # all_step_source_real_delta = []
    # all_step_target_real_delta = []
    # all_step_direct_vel = []
    # all_step_smooth_direct_vel = []
    # for idx, batch in enumerate(train_loader):
    #     if idx % args.len_lb != 0:
    #         continue
    #
    #     print(f"{idx} \n")
    #     obs = batch["observation"]
    #
    #     # for image_g in obs[1][0]:
    #     #     cv2.imshow("asdf", image_g.permute(1, 2, 0).numpy())
    #     # key = cv2.waitKey(1)
    #     # if key == ord("q"):
    #     #     break
    #
    #     all_step_gripper.append(batch["traj"]["gripper"]["obs"][0, -1:, -1:])
    #
    #     all_step_source_pose.append(batch["traj"]["source_pos_quat"]["obs"][0, -1:, :])
    #     all_step_target_pose.append(batch["traj"]["target_pos_quat"]["obs"][0, -1:, :])
    #
    #     all_step_source_real_delta.append(batch["traj"]["source_real_delta"]["obs"][0, -1:, :])
    #     all_step_target_real_delta.append(batch["traj"]["target_real_delta"]["obs"][0, -1:, :])
    #
    #     all_step_direct_vel.append(batch["traj"]["direct_vel"]["obs"][0, -1:, :])
    #     all_step_smooth_direct_vel.append(batch["smooth_traj"]["direct_vel"]["obs"][0, -1:, :])
    #
    #     all_step_source_pos_ori.append(batch["traj"]["source_glb_pos_ori"]["obs"][0, -1:, :])
    #     all_step_target_pos_ori.append(batch["traj"]["target_glb_pos_ori"]["obs"][0, -1:, :])
    #
    # pm = torch.concatenate(all_step_delta, dim=0).detach().cpu().numpy()
    # # pm1 = torch.concatenate(all_step_delta1, dim=0).detach().cpu().numpy()
    # # print(np.sum((pm - pm1)**2))
    # pmr = torch.concatenate(all_step_smooth_delta, dim=0).detach().cpu().numpy()
    # # pmr1 = torch.concatenate(all_step_smooth_delta1, dim=0).detach().cpu().numpy()
    # # print(np.sum((pmr - pmr1)**2))
    # all_step_smooth_pose = torch.concatenate(all_step_smooth_pose, dim=0).detach().cpu().numpy()
    # all_step_pose = torch.concatenate(all_step_source_pose, dim=0).detach().cpu().numpy()
    # all_step_glb_pos_ori = torch.concatenate(all_step_glb_pos_ori, dim=0).detach().cpu().numpy()
    # all_step_smooth_glb_pos_ori = torch.concatenate(all_step_smooth_glb_pos_ori, dim=0).detach().cpu().numpy()
    # all_step_direct_vel = torch.concatenate(all_step_direct_vel, dim=0).detach().cpu().numpy()
    # all_step_smooth_direct_vel = torch.concatenate(all_step_smooth_direct_vel,
    #                                                       dim=0).detach().cpu().numpy()
    # print(np.mean(pm, axis=0))
    # print(np.std(pm, axis=0))
    # print(np.max(pm, axis=0))
    # print(np.min(pm, axis=0))
    #
    # print(np.mean(pmr, axis=0))
    # print(np.std(pmr, axis=0))
    # print(np.max(pmr, axis=0))
    # print(np.min(pmr, axis=0))
    #
    # print(np.mean(all_step_glb_pos_ori, axis=0))
    # print(np.std(all_step_glb_pos_ori, axis=0))
    # print(np.max(all_step_glb_pos_ori, axis=0))
    # print(np.min(all_step_glb_pos_ori, axis=0))
    #
    # print(np.mean(all_step_smooth_glb_pos_ori, axis=0))
    # print(np.std(all_step_smooth_glb_pos_ori, axis=0))
    # print(np.max(all_step_smooth_glb_pos_ori, axis=0))
    # print(np.min(all_step_smooth_glb_pos_ori, axis=0))
    #
    # t = np.arange(pm.shape[0])
    #
    # plt.figure()
    # plt.subplot(711)
    # o = np.stack([all_step_pose[:, 0], all_step_smooth_pose[:, 0]], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # o = np.stack([all_step_pose[:, 1], all_step_smooth_pose[:, 1]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(713)
    # o = np.stack([all_step_pose[:, 2], all_step_smooth_pose[:, 2]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(714)
    # o = np.stack([all_step_pose[:, 3], all_step_smooth_pose[:, 3]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(715)
    # o = np.stack([all_step_pose[:, 4], all_step_smooth_pose[:, 4]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(716)
    # o = np.stack([all_step_pose[:, 5], all_step_smooth_pose[:, 5]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(717)
    # o = np.stack([all_step_pose[:, 6], all_step_smooth_pose[:, 6]], axis=1)
    # plt.plot(t, o, '-')
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(711)
    # o = np.stack([pm[:, 0], pmr[:, 0]], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # o = np.stack([pm[:, 1], pmr[:, 1]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(713)
    # o = np.stack([pm[:, 2], pmr[:, 2]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(714)
    # o = np.stack([pm[:, 3], pmr[:, 3]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(715)
    # o = np.stack([pm[:, 4], pmr[:, 4]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(716)
    # o = np.stack([pm[:, 5], pmr[:, 5]], axis=1)
    # plt.plot(t, o, '-')
    # # plt.subplot(717)
    # # o = np.stack([pm[:, 6], pmr[:, 6]], axis=1)
    # # plt.plot(t, o, '-')
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(711)
    # o = np.stack([all_step_glb_pos_ori[:, 0], all_step_smooth_glb_pos_ori[:, 0]], axis=1)
    # plt.plot(t, o, '-', )
    # plt.subplot(712)
    # o = np.stack([all_step_glb_pos_ori[:, 1], all_step_smooth_glb_pos_ori[:, 1]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(713)
    # o = np.stack([all_step_glb_pos_ori[:, 2], all_step_smooth_glb_pos_ori[:, 2]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(714)
    # o = np.stack([all_step_glb_pos_ori[:, 3], all_step_smooth_glb_pos_ori[:, 3]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(715)
    # o = np.stack([all_step_glb_pos_ori[:, 4], all_step_smooth_glb_pos_ori[:, 4]], axis=1)
    # plt.plot(t, o, '-')
    # plt.subplot(716)
    # o = np.stack([all_step_glb_pos_ori[:, 5], all_step_smooth_glb_pos_ori[:, 5]], axis=1)
    # plt.plot(t, o, '-')
    # # plt.subplot(717)
    # # o = np.stack([all_step_glb_pos_ori[:, 6], all_step_smooth_glb_pos_ori[:, 6]], axis=1)
    # # plt.plot(t, o, '-')
    # plt.show()
    #
    # # plt.figure()
    # # plt.subplot(711)
    # # o = np.stack([all_step_direct_vel[:, 0], all_step_smooth_direct_vel[:, 0]], axis=1)
    # # plt.plot(t, o, '-', )
    # # plt.subplot(712)
    # # o = np.stack([all_step_direct_vel[:, 1], all_step_smooth_direct_vel[:, 1]], axis=1)
    # # plt.plot(t, o, '-')
    # # plt.subplot(713)
    # # o = np.stack([all_step_direct_vel[:, 2], all_step_smooth_direct_vel[:, 2]], axis=1)
    # # plt.plot(t, o, '-')
    # # plt.subplot(714)
    # # o = np.stack([all_step_direct_vel[:, 3], all_step_smooth_direct_vel[:, 3]], axis=1)
    # # plt.plot(t, o, '-')
    # # plt.subplot(715)
    # # o = np.stack([all_step_direct_vel[:, 4], all_step_smooth_direct_vel[:, 4]], axis=1)
    # # plt.plot(t, o, '-')
    # # plt.subplot(716)
    # # o = np.stack([all_step_direct_vel[:, 5], all_step_smooth_direct_vel[:, 5]], axis=1)
    # # plt.plot(t, o, '-')
    # # # plt.subplot(717)
    # # # o = np.stack([all_step_direct_vel[:, 6], all_step_smooth_direct_vel[:, 6]], axis=1)
    # # # plt.plot(t, o, '-')
    # # plt.show()
#######check delta and velocity######################################################################3
