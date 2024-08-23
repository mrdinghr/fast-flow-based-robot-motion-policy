'''compute the stable RFM flow '''
import time

import torch
from tqdm import tqdm


K_ADAP_STEP_PARAM1 = 1.1
K_ADAP_STEP_PARAM2 = 2.


def geodesic(manifold, start_point, end_point, lamda_x):
    shooting_tangent_vec = manifold.logmap(start_point, end_point)
    lamda_x = lamda_x

    def path(t):
        kt = 1 - torch.exp(-lamda_x * t)
        tangent_vecs = torch.einsum("i,...k->...ik", kt, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path


def new_geodesic(manifold, start_point, end_point, lamda_x):
    shooting_tangent_vec = manifold.logmap(end_point, start_point)
    lamda_x = lamda_x

    def path(t):
        kt = torch.exp(-lamda_x * t)
        tangent_vecs = torch.einsum("i,...k->...ik", kt, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(end_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path


# time independent ode solver
'''integration of stable riemannian flow'''
@torch.no_grad()
def projx_integrator(
    manifold, odefunc, z0, lambda_x, ode_steps=10, method="euler", projx=True, local_coords=False, pbar=False, adap_step=False
):
    T = 1.5
    tau1 = 1
    step_fn = {
        "euler": euler_step,
    }[method]

    z_ts = [z0]
    vz_ts = []
    # k_step = 1 - 0.0005 ** (1 / ode_steps)
    # dt = k_step / lambda_tau
    dt = 1 / lambda_x

    if pbar:
        rangelist = tqdm(range(ode_steps))
    else:
        rangelist = range(ode_steps)

    cur_z = z0
    for i in rangelist:
        tau = cur_z[..., -1]
        cur_time = time.time()
        vz_t = odefunc(tau, cur_z)
        # print('vector field ', time.time() - cur_time)
        if not adap_step:
            cur_time = time.time()
            cur_z = step_fn(
                odefunc, cur_z, vz_t, dt, manifold=manifold if local_coords else None
            )
            # print('exp map uses time ', time.time() - cur_time)
            dt = 0.02
        else:
            cur_dt = dt * torch.exp(-0.5 / lambda_x * abs(tau1 - tau))
            cur_z = step_fn(
                odefunc, cur_z, vz_t, cur_dt, manifold=manifold if local_coords else None
            )

        if projx:
            cur_z = manifold.projx(cur_z)
        vz_ts.append(vz_t)
        z_ts.append(cur_z)
    return torch.stack(z_ts), torch.stack(vz_ts)


# todo runge-kutta method implementation
@torch.no_grad()
def rk_projx_integrator(
    manifold, odefunc, x0, lambda_tau, ode_steps=10, method="euler", projx=True, local_coords=False, pbar=False, adap_step=False
):
    T = 3.
    tau0 = 0.
    tau1 = 1.
    t = torch.linspace(0, T, ode_steps + 1).to(x0)
    step_fn = {
        "euler": euler_step,
    }[method]

    xts = [x0]
    vts = []

    xt = x0
    dt = T / ode_steps
    t0s = t[:-1]

    if pbar:
        t0s = tqdm(t0s)

    for t0, t1 in zip(t0s, t[1:]):
        tau = tau1 + (tau0 - tau1) * torch.exp(-lambda_tau * t0)

        if len(tau.shape) == 1:
            vt = odefunc(tau.unsqueeze(0), xt)
        else:
            vt = odefunc(tau, xt)

        if not adap_step:
            xt = step_fn(xt, vt, dt, manifold=manifold if local_coords else None)
        else:
            # k_dt = K_ADAP_STEP_PARAM1 * torch.exp(-K_ADAP_STEP_PARAM2 * t0)
            k_dt = 1.5 - t0 / T
            xt = step_fn(xt, vt, dt * k_dt, manifold=manifold if local_coords else None)

        if projx:
            xt = manifold.projx(xt)
        vts.append(vt)
        xts.append(xt)
    return torch.stack(xts), torch.stack(vts)


# euler method for ODE solving
def euler_step(odefunc, xt, vt, dt, manifold=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt


# integration on learned stable vector field with own writen exp map
# but attention: this is only for Real Robot data with prediction horizon 16
def fast_projx_integrator_robot_data(odefunc, z0, lamda_x, odesteps):
    dt = 1 / lamda_x
    cur_z = z0
    cur_x = z0[:, :-1]
    cur_tau = z0[:, -1:]
    for i in range(odesteps):
        tau = cur_z[..., -1]
        cur_time = time.time()
        vz_t = odefunc(tau, cur_z)
        print('vector field time ', time.time() - cur_time)
        cur_time = time.time()
        vx_t = vz_t[:, :-1]
        vtau_t = vz_t[:, -1:]
        cur_x = fast_exp_map(cur_x, vx_t * dt)
        cur_tau += vtau_t * dt
        cur_z = torch.hstack((cur_x, cur_tau))
        dt = 0.02
        print('exp map uses ', time.time() - cur_time)
    return cur_z


# own writen code of exp map for Real Robot data manifold
# only for product manifold [Euclidean(3), Sphere(4), Euclidean(1)] * 16
def fast_exp_map(x, u):
    x = x.reshape((16, 8))
    u = u.reshape((16, 8))
    x_quat = x[:, 3:7]
    u_quat = u[:, 3:7]
    norm_u_quat = u_quat.norm(dim=-1).unsqueeze(dim=-1)
    exp = x_quat * torch.cos(norm_u_quat) + u_quat * torch.sin(norm_u_quat) / norm_u_quat
    x[:, 3:7] = exp
    x[:, :3] += u[:, :3]
    x[:, -1] += u[:, -1]
    return x.reshape((1, 128))

