import numpy as np
from scipy.interpolate import splrep, BSpline

"""
DEFINE QUATERNION COMPUTATIONS AND MAPPINGS
the logarithmic and exponential maps for quaternion
    q_log_map(p, base=None)
    q_exp_map(p, base=None)
    q_parallel_transport(p, g, h)
and some basic calculations
    q_mul(q1, q2)
    q_inverse(q)
    q_div(q1, q2)
    q_norm_squared(q)
    q_norm(q)
    q_to_rotation_matrix(q)
    q_to_quaternion_matrix(q)
    q_to_euler(q)
    q_from_rot_mat(rot_mat)
    q_from_euler(euler)
    q_to_intrinsic_xyz(q)
    q_from_intrinsic_xyz(euler)
"""


def q_exp_map(v, base=None):
    """
    The exponential quaternion map maps v from the tangent space at base to q on the manifold
    S^3. See Table 2.1 in reference:
    [7] "Programming by Demonstration on Riemannian Manifolds", M.J.A. Zeestraten, 2018.

    Parameters
    ----------
    :param v: np,array of shape (3, N)

    Optional parameters
    -------------------
    :param base: np,array of shape (4,), the base quaternion. If None the neutral element [1, 0, 0, 0] is used.

    Returns
    -------
    :return q: np.array of shape (4, N), the N quaternions corresponding to the N vectors in v
    """
    v_2d = v.reshape((3, 1)) if len(v.shape) == 1 else v
    if base is None:
        norm_v = np.sqrt(np.sum(v_2d ** 2, 0))
        q = np.append(np.ones((1, v_2d.shape[1])), np.zeros((3, v_2d.shape[1])), 0)
        non_0 = np.where(norm_v > 0)[0]
        q[:, non_0] = np.append(
            np.cos(norm_v[non_0]).reshape((1, non_0.shape[0])),
            np.tile(np.sin(norm_v[non_0]) / norm_v[non_0], (3, 1)) * v_2d[:, non_0],
            0,
        )
        return q.reshape(4) if len(v.shape) == 1 else q
    else:
        return q_mul(base, q_exp_map(v))


def q_log_map(q, base=None):
    """
    The logarithmic quaternion map maps q from the manifold S^3 to v in the tangent space at base. See Table 2.1 in [7]

    Parameters
    ----------
    :param q: np,array of shape (4, N), N quaternions

    Optional parameters
    -------------------
    :param base: np,array of shape (4,), the base quaternion. If None the neutral element [1, 0, 0, 0] is used.

    Returns
    -------
    :return v: np.array of shape (3, N), the N vectors in tangent space corresponding to quaternions q
    """
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    if base is None:
        norm_q = np.sqrt(np.sum(q_2d[1:, :] ** 2, 0))
        # to avoid numeric errors for norm_q
        non_0 = np.where((norm_q > 0) * (np.abs(q_2d[0, :]) <= 1))[0]
        q_non_singular = q_2d[:, non_0]
        acos = np.arccos(q_non_singular[0, :])
        # this is *critical* for ensuring q and -q maps are the same rotation
        acos[np.where(q_non_singular[0, :] < 0)] += -np.pi
        v = np.zeros((3, q_2d.shape[1]))
        v[:, non_0] = q_non_singular[1:, :] * np.tile(acos / norm_q[non_0], (3, 1))
        if len(q.shape) == 1:
            return v.reshape(3)
        return v
    else:
        return q_log_map(q_mul(q_inverse(base), q))




def q_parallel_transport(p_g, g, h):
    """
    Transport p in tangent space at g to tangent space at h. According to (2.11)--(2.13) in [7].

    Parameters
    ----------
    :param p_g: np.array of shape (3,) point in tangent space
    :param g: np.array of shape (4,) quaternion
    :param h: np.array of shape (4,) quaternion

    Returns
    -------
    :return p_h: np.array of shape (3,) the point p_g in tangent space at h
    """
    R_e_g = q_to_quaternion_matrix(g)
    R_h_e = q_to_quaternion_matrix(h).T
    B = np.append(np.zeros((3, 1)), np.eye(3), 1).T
    log_g_h = q_log_map(h, base=g)
    m = np.linalg.norm(log_g_h)
    if m < 1e-10:
        return p_g
    u = R_e_g.dot(np.append(0, log_g_h / m)).reshape((4, 1))
    R_g_h = np.eye(4) - np.sin(m) * g.reshape((4, 1)).dot(u.T) + (np.cos(m) - 1) * u.dot(u.T)
    A_g_h = B.T.dot(R_h_e).dot(R_g_h).dot(R_e_g).dot(B)
    return A_g_h.dot(p_g)


def q_mul(q1, q2):
    return q_to_quaternion_matrix(q1).dot(q2)


def q_inverse(q):
    w0, x0, y0, z0 = q
    return np.array([w0, -x0, -y0, -z0]) / q_norm_squared(q)


def q_div(q1, q2):
    return q_mul(q1, q_inverse(q2))


def q_norm_squared(q):
    return np.sum(q ** 2)


def q_norm(q):
    return np.sqrt(q_norm_squared(q))


def q_to_rotation_matrix(q):
    """
    Computes rotation matrix out of the quaternion.

    Parameters
    ----------
    :param q: np.array of shape (4,), the quaternion

    Returns
    -------
    :return rot_mat: np.array of shape (3, 3)
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)],
        ]
    )


def q_to_quaternion_matrix(q):
    """
    Computes quaternion matrix out of the quaternion.

    Parameters
    ----------
    :param q: np.array of shape (4,), the quaternion

    Returns
    -------
    :return quat_mat: np.array of shape (4, 4)
    """
    return np.array(
        [[q[0], -q[1], -q[2], -q[3]], [q[1], q[0], -q[3], q[2]], [q[2], q[3], q[0], -q[1]], [q[3], -q[2], q[1], q[0]]]
    )


def q_to_euler(q):
    """
    Computes euler angles out of the quaternion. Format XYZ (roll, pitch, yaw)

    Parameters
    ----------
    :param q: np.array of shape (4, N) or (4,), the quaternion

    Returns
    -------
    :return euler: np.array of shape (3, N) or (3,)
    """
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    w, x, y, z = q_2d[0, :], q_2d[1, :], q_2d[2, :], q_2d[3, :]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    euler = np.stack([roll, pitch, yaw], axis=0)
    euler = (
        euler.reshape(
            3,
        )
        if euler.shape[1] == 1
        else euler
    )
    return euler


def q_from_rot_mat(rot_mat):
    """
    Computes the quaternion out of a given rotation matrix

    Parameters
    ----------
    :param rot_mat:  np.array of shape (3, 3)

    Returns
    -------
    :return q: np.array of shape (4,), quaternion corresponding to the rotation matrix rot_mat
    """
    qs = min(np.sqrt(np.trace(rot_mat) + 1) / 2.0, 1.0)
    kx = rot_mat[2, 1] - rot_mat[1, 2]  # Oz - Ay
    ky = rot_mat[0, 2] - rot_mat[2, 0]  # Ax - Nz
    kz = rot_mat[1, 0] - rot_mat[0, 1]  # Ny - Ox
    if (rot_mat[0, 0] >= rot_mat[1, 1]) and (rot_mat[0, 0] >= rot_mat[2, 2]):
        kx1 = rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2] + 1  # Nx - Oy - Az + 1
        ky1 = rot_mat[1, 0] + rot_mat[0, 1]  # Ny + Ox
        kz1 = rot_mat[2, 0] + rot_mat[0, 2]  # Nz + Ax
        add = kx >= 0
    elif rot_mat[1, 1] >= rot_mat[2, 2]:
        kx1 = rot_mat[1, 0] + rot_mat[0, 1]  # Ny + Ox
        ky1 = rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2] + 1  # Oy - Nx - Az + 1
        kz1 = rot_mat[2, 1] + rot_mat[1, 2]  # Oz + Ay
        add = ky >= 0
    else:
        kx1 = rot_mat[2, 0] + rot_mat[0, 2]  # Nz + Ax
        ky1 = rot_mat[2, 1] + rot_mat[1, 2]  # Oz + Ay
        kz1 = rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1] + 1  # Az - Nx - Oy + 1
        add = kz >= 0
    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1
    nm = np.linalg.norm(np.array([kx, ky, kz]))
    if nm == 0:
        q = np.array([1, 0, 0, 0])
    else:
        s = np.sqrt(1 - qs ** 2) / nm
        qv = s * np.array([kx, ky, kz])
        q = np.append(qs, qv)
    return q


def q_from_euler(euler):
    """
    Computes quaternion out of the euler angles.

    Parameters
    ----------
    :param euler: np.array of shape (3, N) or (3,). Euler angles XYZ (roll, pitch, yaw)

    Returns
    -------
    :return q: np.array of shape (4, N) or (4,)
    """
    euler_2d = euler.reshape((3, 1)) if len(euler.shape) == 1 else euler
    roll, pitch, yaw = euler_2d[0, :], euler_2d[1, :], euler_2d[2, :]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.stack([w, x, y, z], axis=0)
    q = (
        q.reshape(
            4,
        )
        if q.shape[1] == 1
        else q
    )
    return q


def q_from_intrinsic_xyz(euler):
    """
    Convert euler angle as intrinsic rotations x-y-z to quaternion.
    Note: Euler angles in "q_from_euler()" is defined as extrinsic rotations x-y-z (roll, pitch, yaw),
    or equivalently to intrinsic rotations z-y-x.

    Parameters
    ----------
    :param euler: np.array of shape (3, N) or (3,). Euler angle as intrinsic rotations x-y-z.

    Returns
    -------
    :return q: np.array of shape (4, N) or (4,)
    """
    euler_2d = euler.reshape((3, 1)) if len(euler.shape) == 1 else euler
    angle_x, angle_y, angle_z = euler_2d[0, :], euler_2d[1, :], euler_2d[2, :]
    cx = np.cos(angle_x * 0.5)
    sx = np.sin(angle_x * 0.5)
    cy = np.cos(angle_y * 0.5)
    sy = np.sin(angle_y * 0.5)
    cz = np.cos(angle_z * 0.5)
    sz = np.sin(angle_z * 0.5)

    w = cx * cy * cz - sx * sy * sz
    x = cx * sy * sz + sx * cy * cz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    q = np.stack([w, x, y, z], axis=0)
    q = (
        q.reshape(
            4,
        )
        if q.shape[1] == 1
        else q
    )
    return q


def q_to_intrinsic_xyz(q):
    """
    Computes euler angles out of the intrinsic xyz rotations.

    Parameters
    ----------
    :param q: np.array of shape (4, N) or (4,), the quaternion.

    Returns
    -------
    :return euler: np.array of shape (3, N) or (3,)
    """
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    w, x, y, z = q_2d[0, :], q_2d[1, :], q_2d[2, :], q_2d[3, :]
    t0 = 2.0 * (w * x - y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    angle_x = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y + z * x), -1.0, 1.0)
    angle_y = np.arcsin(t2)
    t3 = 2.0 * (w * z - x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    angle_z = np.arctan2(t3, t4)
    euler = np.stack([angle_x, angle_y, angle_z], axis=0)
    euler = (
        euler.reshape(
            3,
        )
        if euler.shape[1] == 1
        else euler
    )
    return euler


def log_map(pos, base):
    delta_p = pos[:3] - base[:3]
    delta_o = q_log_map(pos[3:], base[3:])
    return np.concatenate([delta_p, delta_o])


def exp_map(delta, base):
    pos_p = delta[:3] + base[:3]
    pos_o = q_exp_map(delta[3:], base[3:])
    return np.concatenate([pos_p, pos_o])


def exp_map_seq(delta, base):
    pos_p = delta[:, :3] + base[None, :3]
    pos_o = q_exp_map(delta[:, 3:].transpose(1, 0), base[3:]).transpose(1, 0)
    return np.concatenate([pos_p, pos_o], axis=1)


def log_map_seq(pos, base):
    delta_p = pos[:, :3] - base[None, :3]
    delta_o = q_log_map(pos[:, 3:].transpose(1, 0), base[3:]).transpose(1, 0)
    return np.concatenate([delta_p, delta_o], axis=1)


def recover_pose_from_quat_real_delta(future_vel_seq: np.ndarray, base: np.ndarray):
    """
    base: [7,]
    future_pose_seq: [10, 7]

    """
    recover_pose = np.zeros([future_vel_seq.shape[0], 7])
    for i in range(future_vel_seq.shape[0]):
        out = exp_map(future_vel_seq[i, :], base)
        base = out
        recover_pose[i] = out
    return recover_pose


def smooth_traj(pm: np.ndarray, s: tuple) -> (np.ndarray, np.ndarray):
    """
    pm: input trajectory, with shape [N, 7]
    """
    t = np.arange(pm.shape[0])
    pm_delta = log_map_seq(pm.copy(), np.array([0, 0, 0, 0, 1, 0, 0]))
    pmr_delta = []
    for i in range(6):
        x = BSpline(*splrep(t, pm_delta[:, i], s=s[i]))(t)
        pmr_delta.append(x)
    pmr_delta = np.stack(pmr_delta, axis=1)
    pmr = exp_map_seq(pmr_delta, np.array([0, 0, 0, 0, 1, 0, 0]))
    return pmr, pm_delta, pmr_delta


if __name__ == "__main__":
    # q1 = np.array([1,2,3,4])/np.linalg.norm([1,2,3,4],2)
    # q2 = np.array([2,3,4,5])/np.linalg.norm([2,3,4,5],2)
    # q3 = np.array([3,4,5,6])/np.linalg.norm([3,4,5,6],2)
    # v12 = q_log_map(q2, q1)
    # v23 = q_log_map(q3, q2)
    #
    # v1 = q_log_map(q1)
    # v2 = q_log_map(q2)
    # v3 = q_log_map(q3)
    # v12_ = v2 - v1
    # v23_ = v3 - v2
    # print(v12 - v12_)
    # print(v23 - v23_)


    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # Represent initial and final orientations as quaternions
    q_initial = np.array([
                0.012322934256336735,
                0.9786744566129287,
                0.19673342223481813,
                -0.057796271671211055
            ])
    q_final = np.array([
                0.005934300385392491,
                0.9879155098505336,
                0.14772539998482817,
                -0.04652894347449135
            ])

    # q_initial = np.array([0., 0., 1., 0.])
    # q_final = np.array([0., 1., 0., 0.])

    out1 = q_log_map(q_final, q_initial)

    out1_ = q_log_map(q_final) - q_log_map(q_initial)

    q_initial = np.concatenate([q_initial[1:], q_initial[0:1]])
    q_final = np.concatenate([q_final[1:], q_final[0:1]])
    q_initial_rot = R.from_quat(q_initial)
    q_final_rot = R.from_quat(q_final)
    q_rel = q_final_rot.inv() * q_initial_rot  # ? ?
    print(q_final_rot.as_matrix())
    print(q_final_rot.inv().as_matrix())
    print(q_initial_rot.as_matrix())
    q_rel = q_rel.as_quat()

    # Convert relative quaternion to axis-angle representation
    angle = 2 * np.arccos(q_rel[-1])  # Angle of rotation
    axis = q_rel[:-1] / np.sqrt(1 - q_rel[-1] ** 2)
    out2 = angle * axis
    print(out1 / out2)
    del()

    ## [0.99867253 0.00698209 0.01221408 0.04955053]
    ## [-0.00698209 -0.01221408 -0.04955053  0.99867252]

    def quat2mat(quaternion):
        import math
        """
        Converts given quaternion to matrix.

        Args:
            quaternion (np.array): (x,y,z,w) vec4 float angles

        Returns:
            np.array: 3x3 rotation matrix
        """
        # awkward semantics for use with numba
        inds = np.array([3, 0, 1, 2])
        q = np.asarray(quaternion).copy().astype(np.float32)[inds]

        n = np.dot(q, q)
        if n < 1e-8:
            return np.identity(3)
        q *= math.sqrt(2.0 / n)
        q2 = np.outer(q, q)
        return np.array(
            [
                [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
                [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
                [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
            ]
        )

    q_init = quat2mat(q_initial)
    q_fin = quat2mat(q_final)
    # goal_orientation = np.dot(rotation_mat_error, current_orientation)
    # rotation_mat_error = goal_orientation * current_orientation.inv()
    q_err = np.dot(q_fin.transpose(), q_init)


    def mat2quat(rmat):
        """
        Converts given rotation matrix to quaternion.

        Args:
            rmat (np.array): 3x3 rotation matrix

        Returns:
            np.array: (x,y,z,w) float quaternion angles
        """
        M = np.asarray(rmat).astype(np.float32)[:3, :3]

        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
        if q1[0] < 0.0:
            np.negative(q1, q1)
        inds = np.array([1, 2, 3, 0])
        return q1[inds]

    q_err = mat2quat(q_err)


    def quat2axisangle(quat):
        import math
        """
        Converts quaternion to axis-angle format.
        Returns a unit vector direction scaled by its angle in radians.

        Args:
            quat (np.array): (x,y,z,w) vec4 float angles

        Returns:
            np.array: (ax,ay,az) axis-angle exponential coordinates
        """
        # clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    delta = quat2axisangle(q_err)

    print(delta / out1)


    def axisangle2quat(vec):
        import math
        """
        Converts scaled axis-angle to quat.

        Args:
            vec (np.array): (ax,ay,az) axis-angle exponential coordinates

        Returns:
            np.array: (x,y,z,w) vec4 float angles
        """
        # Grab angle
        angle = np.linalg.norm(vec)

        # handle zero-rotation case
        if math.isclose(angle, 0.0):
            return np.array([0.0, 0.0, 0.0, 1.0])

        # make sure that axis is a unit vector
        axis = vec / angle

        q = np.zeros(4)
        q[3] = np.cos(angle / 2.0)
        q[:3] = axis * np.sin(angle / 2.0)
        return q


    quat_error = axisangle2quat(delta)
    rotation_mat_error = quat2mat(quat_error)
    goal_orientation = np.dot(rotation_mat_error, q_init)
    print((q_fin - goal_orientation)/goal_orientation)

