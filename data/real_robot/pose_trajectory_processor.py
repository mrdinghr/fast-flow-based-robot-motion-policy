import quaternion
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
import os
from tqdm import tqdm

@dataclass
class Translation:
    x: float
    y: float
    z: float

    def __post_init__(self):
        if isinstance(self.x, np.ndarray):
            self.x = self.x[0]
        if isinstance(self.y, np.ndarray):
            self.y = self.y[0]
        if isinstance(self.z, np.ndarray):
            self.z = self.z[0]

    @classmethod
    def origin(cls):
        return cls(0, 0, 0)

    def __str__(self):
        return f"[{self.x:.2f}, {self.y:.2f}, {self.z:.2f}]"

    def vec(self):
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_vec(cls, vec):
        return cls(vec[0], vec[1], vec[2])

@dataclass
class Displacement:
    dx: float
    dy: float
    dz: float

    def __post_init__(self):
        if isinstance(self.dx, np.ndarray):
            self.dx = self.dx[0]
        if isinstance(self.dy, np.ndarray):
            self.dy = self.dy[0]
        if isinstance(self.dz, np.ndarray):
            self.dz = self.dz[0]

    @classmethod
    def origin(cls):
        return cls(0, 0, 0)

    def __str__(self):
        return f"[{self.dx:.2f}, {self.dy:.2f}, {self.dz:.2f}]"

    def vec(self):
        return np.array([self.dx, self.dy, self.dz])

    @classmethod
    def from_vec(cls, vec):
        return cls(vec[0], vec[1], vec[2])

@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self):
        if isinstance(self.x, np.ndarray):
            self.x = self.x[0]
        if isinstance(self.y, np.ndarray):
            self.y = self.y[0]
        if isinstance(self.z, np.ndarray):
            self.z = self.z[0]
        if isinstance(self.w, np.ndarray):
            self.w = self.w[0]

    @classmethod
    def origin(cls):
        return cls(1, 0, 0, 0)

    def __post_init__(self):
        norm = (self.x**2 + self.y**2 + self.z ** 2 + self.w**2) ** .5
        if abs(norm - 1.0) > 1e-5:
            raise ValueError(f"Quaternion not unit norm: {norm:.5f}!")

    def vec_xyzw(self):
        return np.array([self.x, self.y, self.z, self.w])

    def vec_wxyz(self):
        return np.array([self.w, self.x, self.y, self.z])

    def vec(self, quat_order="wxyz"):
        assert quat_order in ["xyzw", "wxyz"]
        # default behavior as in the quaternion class of tami_lfd
        if quat_order == "wxyz":
            return self.vec_wxyz()
        else:
            return self.vec_xyzw()

    def __str__(self):
        return f"[({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), {self.w:.2f}]"

    @classmethod
    def from_vec_xyzw(cls, vec):
        return cls(vec[0], vec[1], vec[2], vec[3])

    @classmethod
    def from_vec_wxyz(cls, vec):
        return cls(vec[1], vec[2], vec[3], vec[0])

    @classmethod
    def from_vec(cls, vec, quat_order):
        assert quat_order in ["xyzw", "wxyz"]
        if quat_order == "wxyz":
            return cls.from_vec_wxyz(vec)
        else:
            return cls.from_vec_xyzw(vec)

    def flip_sign(self):
        self.x, self.y, self.z, self.w = -self.x, -self.y, -self.z, -self.w

@dataclass
class QuaternionVelocity:
    """ The S3 velocity in the tangent space of base """
    base: Optional[Quaternion]
    vel: Displacement

    @classmethod
    def from_vec(cls, vel, base=None, quat_order="wxyz"):
        return cls(Quaternion.from_vec(base, quat_order), Displacement.from_vec(vel))

    def vec(self):
        return self.vel.vec()

@dataclass
class Pose:
    position: Translation
    orientation: Quaternion
    base_frame: str = ""

    @classmethod
    def origin(cls):
        pos = Translation.origin()
        quat = Quaternion.origin()
        return cls(pos, quat)

    def __str__(self):
        return f"{self.position.__str__()} | {self.orientation.__str__()}"

    @classmethod
    def from_vec(cls, vec, quat_order="wxyz"):
        trans = Translation.from_vec(vec[:3])
        quat = Quaternion.from_vec(vec[3:], quat_order)
        return cls(trans, quat)

    def vec(self, quat_order="wxyz") -> np.ndarray:
        return np.concatenate((self.position.vec(), self.orientation.vec(quat_order=quat_order)), axis=0)

    def flip_quaternion_sign(self):
        self.orientation.flip_sign()

@dataclass
class PoseVelocity:
    pos_vel: Displacement
    quat_vel: QuaternionVelocity

    @classmethod
    def from_vec(cls, vec, quat_base=None, quat_order="wxyz"):
        pos_vel = Displacement.from_vec(vec[:3])
        quat_vel = QuaternionVelocity.from_vec(vec[3:], quat_base, quat_order)
        return cls(pos_vel, quat_vel)

    def vec(self):
        return np.concatenate((self.pos_vel.vec(), self.quat_vel.vec()), axis=0)

@dataclass
class RobotTrajectory:
    poses: List[Pose]
    grippers: List[List[float]]  # [left_finger, right_finger]
    time_stamps: List[float]

    def __post_init__(self):
        if len(self.poses) != len(self.time_stamps) or len(self.grippers) != len(self.time_stamps):
            raise ValueError("Length of poses and time-stamps does not equal")

    def pose_matrix(self, quat_order="wxyz"):
        """ Returns poses and gripper state in compact format:
            (pos), (quat), (gripper_left/right): xyz, wxyz, lr"""
        assert quat_order in ["xyzw", "wxyz"]
        N = len(self.poses)
        pose_matrix = np.zeros((N, 9))
        for i in range(N):
            q_vec = self.poses[i].orientation.vec_wxyz() if quat_order == "wxyz" else \
                self.poses[i].orientation.vec_xyzw()
            pose_matrix[i, :7] = np.concatenate((self.poses[i].position.vec(), q_vec), axis=0)
            pose_matrix[i, 7:] = np.array(self.grippers[i])
        return pose_matrix, np.array(self.time_stamps)

    @classmethod
    def from_matrix(cls, pm, time_steps, quat_order="wxyz"):
        N = pm.shape[0]
        assert pm.shape[1] == 9
        poses = []
        grippers = []
        time_steps_list = []
        for i in range(N):
            pose = Pose.from_vec(pm[i, :7], quat_order)
            time_step = time_steps[i]
            gripper = pm[i, 7:]
            poses.append(pose)
            grippers.append([gripper[0], gripper[1]])
            time_steps_list.append(time_step)
        return cls(poses, grippers, time_steps)

    def _to_dict(self, quat_order="wxyz") -> Dict:
        assert len(self.poses) == len(self.time_stamps)
        assert len(self.grippers) == len(self.time_stamps)
        N = len(self.poses)

        d = {}
        for i in range(N):
            d_loc = {}
            d_loc['quat_order'] = quat_order
            pose_vec = self.poses[i].vec(quat_order=quat_order).tolist()
            d_loc['pose'] = pose_vec
            d_loc['gripper'] = self.grippers[i]
            d_loc['time'] = self.time_stamps[i]
            d[i] = d_loc
        return d


    def save_to_file(self, file_name):
        d = self._to_dict()
        with open(file_name, 'w') as f:
            json.dump(d, f, indent=4, ensure_ascii=False)

        print(f"Saved trajectory to {file_name}")

    @classmethod
    def from_path(cls, path):
        assert os.path.exists(path)
        with open(path) as f:
            dat = json.load(f)
        return cls.from_dict(dat)

    @classmethod
    def from_dict(cls, d):
        N = len(d.keys()) - 1
        pose_list = []
        gripper_list = []
        time_stamps = []

        for i in range(N):
            dat = d[str(i)]
            quat_order = dat['quat_order']
            pose_vec = dat['pose']
            gripper = dat['gripper']
            pose = Pose.from_vec(pose_vec, quat_order=quat_order)
            pose_list.append(pose)
            gripper_list.append(gripper)
            time_stamps.append(dat['time'])

        return cls(pose_list, gripper_list, time_stamps)

@dataclass
class ProcessedRobotTrajectory:
    """ Ordered trajectory with preset sampling time """
    pose_trajectory: RobotTrajectory
    velocity_trajectory: Optional[List[PoseVelocity]]
    sampling_time: float

    def __len__(self):
        return len(self.pose_trajectory.poses)

    def forward_integrate(self, dt: Optional[float]=None, p0: Optional[Pose]=None) -> RobotTrajectory:
        assert len(self.velocity_trajectory) > 0
        if p0 is None:
            p0: Pose = pose_trajectory.poses[0]

        N = len(self.velocity_trajectory)
        if dt is None:
            dt = self.sampling_time

        pose_list = []
        time_list = []
        pose_list.append(p0)
        time_list.append(0.0)

        p_curr = copy.copy(p0)

        for i in range(N):
            pose_vel: PoseVelocity = self.velocity_trajectory[i]
            p_new = copy.copy(p_curr)
            new_pos = p_new.position.vec() + dt * pose_vel.pos_vel.vec()
            p_new.position = Translation.from_vec(new_pos)

            q_vel: QuaternionVelocity = pose_vel.quat_vel
            q_vel_base = q_vel.base.vec_wxyz()
            q_curr = p_new.orientation.vec_wxyz()
            q_vel_curr = quaternion.q_parallel_transport(q_vel.vel.vec(), q_vel_base, q_curr)
            q_new_vec = quaternion.q_exp_map(dt* q_vel_curr, q_curr)
            p_new.orientation = Quaternion.from_vec_wxyz(q_new_vec)

            p_curr = p_new

            pose_list.append(p_curr)
            time_list.append((i+1)*dt)

        return RobotTrajectory(pose_list, self.pose_trajectory.grippers, time_list)

    def _to_dict(self, quat_order="wxyz") -> Dict:
        has_pose = len(self.pose_trajectory.poses) > 0
        has_vel = len(self.velocity_trajectory) > 0
        if not has_pose and not has_vel:
            raise ValueError("No data to save!")

        N = len(self.pose_trajectory.poses) if has_pose else len(self.velocity_trajectory)

        d = {"sampling_time": self.sampling_time}
        for i in range(N):
            d_loc = {}
            d_loc['quat_order'] = quat_order
            if has_pose:
                pose_vec = self.pose_trajectory.poses[i].vec(quat_order=quat_order).tolist()
                d_loc['pose'] = pose_vec
                d_loc['gripper'] = self.pose_trajectory.grippers[i]
            if has_vel and i < N-1:  # we have one less velocity
                vel_vec = self.velocity_trajectory[i].vec().tolist()
                d_loc['vel'] = vel_vec
            d[i] = d_loc
        return d

    def save_to_file(self, file_name):
        d = self._to_dict()
        split_name = file_name.split("/")
        split_name[-1] = "resampled_" + split_name[-1]
        file_name_new = "/".join(split_name)
        with open(file_name_new, 'w') as f:
            json.dump(d, f, indent=4, ensure_ascii=False)

        print(f"Saved resampled trajectory to {file_name_new}")

    @classmethod
    def from_dict(cls, d):
        sampling_time = d["sampling_time"]
        N = len(d.keys()) - 1
        pose_list = []
        gripper_list = []
        vel_list = []
        time_stamps = []

        for i in range(N):
            dat = d[str(i)]
            quat_order = dat['quat_order']
            pose_vec = dat['pose']
            gripper = dat['gripper']
            pose = Pose.from_vec(pose_vec, quat_order=quat_order)
            pose_list.append(pose)
            gripper_list.append(gripper)
            time_stamps.append(i*sampling_time)
            if "vel" in dat.keys() and i < N-1:
                vel_vec = dat['vel']
                vel = PoseVelocity.from_vec(vel_vec, quat_order=quat_order, quat_base=pose.orientation.vec(quat_order=quat_order))
                vel_list.append(vel)

        pose_traj = RobotTrajectory(pose_list, gripper_list, time_stamps)
        return cls(pose_traj, vel_list, sampling_time)

    @classmethod
    def from_path(cls, path):
        assert os.path.exists(path)
        with open(path) as f:
            dat = json.load(f)
        return cls.from_dict(dat)


class PoseTrajectoryProcessor:

    def __init__(self):
        pass

    def preprocess_trajectory(self, trajectory: Any) -> RobotTrajectory:
        """ Preprocess the recording to the right format """
        if isinstance(trajectory, RobotTrajectory):
            return trajectory
        elif isinstance(trajectory, Dict):
            return self._preprocess_dict_traj(trajectory)
        else:
            raise NotImplementedError("import trajectory should be Dict or RobotTrajectory")

    def _preprocess_dict_traj(self, traj: Dict) -> RobotTrajectory:
        N = len(traj)
        pose_list = []
        gripper_list = []
        time_stamps = []
        for i in range(N):
            pose_dict = traj[str(i)]
            quat_order = pose_dict['quat_order']
            pose_vec = np.array(pose_dict['pose'])
            gripper = pose_dict['gripper']
            if i == 0:
                t0 = pose_dict['time']
            time = pose_dict['time'] - t0

            pose = Pose.from_vec(pose_vec, quat_order=quat_order)
            pose_list.append(pose)
            time_stamps.append(time)
            gripper_list.append(gripper)

        return RobotTrajectory(pose_list, gripper_list, time_stamps)


    def process_pose_trajectory(self, trajectory: RobotTrajectory, sampling_time: float) -> ProcessedRobotTrajectory:
        print("Interpolating trajectory ...")
        time_stamps = np.array(trajectory.time_stamps)
        print(f"Average recording frequency: {1.0/np.mean(np.diff(time_stamps)):.1f} Hz")
        final_time_step = time_stamps[-1]
        num_time_steps = int(final_time_step // sampling_time) + 1
        sample_time_stamps = np.array([sampling_time * i for i in range(num_time_steps)])

        pm, _ = trajectory.pose_matrix(quat_order="wxyz")
        N = pm.shape[0]

        # position_interpolation
        tx = np.expand_dims(np.interp(sample_time_stamps, time_stamps, pm[:, 0]), axis=1)
        ty = np.expand_dims(np.interp(sample_time_stamps, time_stamps, pm[:, 1]), axis=1)
        tz = np.expand_dims(np.interp(sample_time_stamps, time_stamps, pm[:, 2]), axis=1)

        # add final position
        tx = np.concatenate((tx, np.array([[pm[-1, 0]]])), axis=0)
        ty = np.concatenate((ty, np.array([[pm[-1, 1]]])), axis=0)
        tz = np.concatenate((tz, np.array([[pm[-1, 2]]])), axis=0)

        p_interp = np.concatenate((tx, ty, tz), axis=1)

        # gripper interpolation
        gl = np.expand_dims(np.interp(sample_time_stamps, time_stamps, pm[:, 7]), axis=1)
        gr = np.expand_dims(np.interp(sample_time_stamps, time_stamps, pm[:, 8]), axis=1)
        gl = np.concatenate((gl, np.array([[pm[-1, 7]]])), axis=0)
        gr = np.concatenate((gr, np.array([[pm[-1, 8]]])), axis=0)
        g_interp = np.concatenate((gl, gr), axis=1)

        # An exhaustive implementation of quaternion interpolation
        q_interp = np.zeros((num_time_steps + 1, 4))
        pbar = tqdm(range(num_time_steps), desc="Resampling quaternion", leave=False)
        for i in pbar:
            # find closes sampling point
            curr_interpolation_time = i * sampling_time
            demo_ix = np.argmin((time_stamps - curr_interpolation_time)**2)

            # Choose this quaternion as the basis for smooth local interpolation
            q0 = pm[demo_ix, 3:7]
            # generate exponential coordinates in local base
            qm = np.zeros((N, 3))  # last is distance from base
            for ii in range(N):
                q_loc = quaternion.q_log_map(pm[ii, 3:7], q0)
                qm[ii, :] = q_loc

            # quaternion interpolation
            qvx = np.expand_dims(np.interp(sample_time_stamps, time_stamps, qm[:, 0]), axis=1)
            qvy = np.expand_dims(np.interp(sample_time_stamps, time_stamps, qm[:, 1]), axis=1)
            qvz = np.expand_dims(np.interp(sample_time_stamps, time_stamps, qm[:, 2]), axis=1)

            # Map back to manifold
            v_loc = np.array([qvx[i], qvy[i], qvz[i]])
            q_loc = quaternion.q_exp_map(v_loc, q0)[:, 0]
            if i > 0:
                if np.dot(q_loc, q_interp[i-1, :]) < 0.0:
                    q_loc *= -1.0
            q_interp[i, :] = q_loc

        q_interp[num_time_steps, :] = pm[-1, 3:7]

        # generate absolute velocities
        pose_velocity = np.zeros((num_time_steps+1, 6))
        pv_list = []
        for i in range(num_time_steps):
            pose_velocity[i, :3] = (p_interp[i+1, :] - p_interp[i, :])/sampling_time
            pose_velocity[i, 3:] = quaternion.q_log_map(q_interp[i+1, :], q_interp[i, :])/sampling_time
            pv = PoseVelocity.from_vec(pose_velocity[i, :], quat_base=q_interp[i, :])
            pv_list.append(pv)

        robot_interp = np.concatenate((p_interp, q_interp, g_interp), axis=1)

        sample_time_stamps = np.concatenate((sample_time_stamps, np.array([num_time_steps * sampling_time])), axis=0)

        new_pose_traj = RobotTrajectory.from_matrix(robot_interp, sample_time_stamps)
        processed_traj = ProcessedRobotTrajectory(new_pose_traj, pv_list, sampling_time=sampling_time)

        print("... done")
        return processed_traj


if __name__ == "__main__":

    file_names = ["C:/Users/kup2rng/Downloads/traj_recordings/demo_02_13_2024_14_05_50.json",
                  "C:/Users/kup2rng/Downloads/traj_recordings/demo_02_13_2024_14_07_15.json"]
    file_name_orig = file_names[0]
    file_name_orig = "/tmp/demo_02_20_2024_14_11_09.json"
    with open(file_name_orig) as f:
        pose_trajectory = json.load(f)
        print("Trajectory loaded ...")


    processor = PoseTrajectoryProcessor()
    pose_trajectory = processor.preprocess_trajectory(pose_trajectory)
    resampled_trajectory = processor.process_pose_trajectory(pose_trajectory, sampling_time=0.1)
    resampled_trajectory.save_to_file(file_name_orig)
    fwd_int_traj = resampled_trajectory.forward_integrate()

    pm, t = pose_trajectory.pose_matrix()
    pmr, tr = resampled_trajectory.pose_trajectory.pose_matrix()
    pmfwd, tfwd = fwd_int_traj.pose_matrix()
    plt.figure()
    plt.subplot(211)
    plt.plot(t, pm[:, :3], 'o-')
    plt.plot(tr, pmr[:, :3], 'x--')
    plt.plot(tfwd, pmfwd[:, :3], 'd-')
    plt.subplot(212)
    plt.plot(t, pm[:, 3:], 'o-')
    plt.plot(tr, pmr[:, 3:], 'x--')
    plt.plot(tfwd, pmfwd[:, 3:], 'd-')
    plt.show()

