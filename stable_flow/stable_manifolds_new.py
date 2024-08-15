'''
compute the stable RFM flow
this new one is that tau is sampled, not t

something weird and wrong, when integrate, it should integrate along tau, this may not be possible
'''

import torch
from tqdm import tqdm


def geodesic(manifold, start_point, end_point, lamda_x, lambda_tau):
    shooting_tangent_vec = manifold.logmap(start_point, end_point)
    lamda_x = lamda_x

    def path(tau):
        ktau = 1 - ((1 - tau) ** (lamda_x / lambda_tau))
        tangent_vecs = torch.einsum("i,...k->...ik", ktau, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path


'''integration of stable riemannian flow'''
@torch.no_grad()
def projx_integrator(
    manifold, odefunc, x0, lambda_tau, ode_steps=10, method="euler", projx=True, local_coords=False, pbar=False
):
    t1 = 1.5

    step_fn = {
        "euler": euler_step,
    }[method]

    xts = [x0]
    vts = []

    xt = x0
    dt = t1 / ode_steps
    tau = torch.zeros(1)

    for time_step in tqdm(range(ode_steps)):
        tau = 1 - torch.exp(torch.tensor(-lambda_tau * dt * time_step)).to(x0)
        if len(tau.shape) == 1:
            vt = odefunc(tau.unsqueeze(0), xt)
        else:
            vt = odefunc(tau, xt)
        xt = step_fn(xt, vt, dt, manifold=manifold if local_coords else None)

        if projx:
            xt = manifold.projx(xt)
        vts.append(vt)
        xts.append(xt)
    return torch.stack(xts), torch.stack(vts)


def euler_step(xt, vt, dt, manifold=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt
