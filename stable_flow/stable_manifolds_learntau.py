'''compute the stable RFM flow '''

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
    manifold, odefunc, z0, lambda_tau, ode_steps=10, method="euler", projx=True, local_coords=False, pbar=False, adap_step=False
):
    T = 1.5
    tau1 = 1
    step_fn = {
        "euler": euler_step,
    }[method]

    z_ts = [z0]
    vz_ts = []
    # dt = T / ode_steps
    # dt = 0.05  # for ode step 100
    # dt = 0.07  # best dt
    # dt = 0.25  # best for srfm ode 5 lambda 2.5
    # # dt = 0.12
    k_step = 1 - 0.0005 ** (1 / ode_steps)
    dt = k_step / lambda_tau
    dt = 1 / lambda_tau

    # dt = 1 / lambda_tau

    if pbar:
        rangelist = tqdm(range(ode_steps))
    else:
        rangelist = range(ode_steps)

    cur_z = z0
    for i in rangelist:
        tau = cur_z[..., -1]
        vz_t = odefunc(tau, cur_z)
        # vz_t[..., -1] = -lambda_tau * (tau - tau1)

        # if not adap_step:
        #     xt = step_fn(
        #         odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None
        #     )
        # else:
        #     # k_dt = K_ADAP_STEP_PARAM1 * torch.exp(-K_ADAP_STEP_PARAM2 * t0)
        #     k_dt = 2 - t0 / T
        #     xt = step_fn(xt, vt, dt * k_dt, manifold=manifold if local_coords else None)

        if not adap_step:
            cur_z = step_fn(
                odefunc, cur_z, vz_t, dt, manifold=manifold if local_coords else None
            )
            dt = 0.02
            # if i > ode_steps / 2:
            #     dt = 1 / (lambda_tau * ode_steps)
        else:
            cur_dt = dt * torch.exp(-0.5 / lambda_tau * abs(tau1 - tau))
            cur_z = step_fn(
                odefunc, cur_z, vz_t, cur_dt, manifold=manifold if local_coords else None
            )
            # if tau < 0.95:
            #     cur_z = step_fn(
            #         odefunc, cur_z, vz_t, dt, manifold=manifold if local_coords else None
            #     )
            # else:
            #     cur_z = step_fn(
            #         odefunc, cur_z, vz_t, dt * 0.1, manifold=manifold if local_coords else None
            #     )

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

        # if t0 < T / 3:
        #     xt = step_fn(xt, vt, dt * 1.3, manifold=manifold if local_coords else None)
        # else:
        #     xt = step_fn(xt, vt, dt * 0.9, manifold=manifold if local_coords else None)

        if projx:
            xt = manifold.projx(xt)
        vts.append(vt)
        xts.append(xt)
    return torch.stack(xts), torch.stack(vts)


def euler_step(odefunc, xt, vt, dt, manifold=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt

