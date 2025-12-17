import numpy as np
from numba import njit, prange

@njit
def limit_magnitude(v, max_val):
    mag = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag > max_val and mag > 0.0:
        inv = max_val / mag
        v[0] *= inv
        v[1] *= inv
        v[2] *= inv


@njit
def wrap_around(pos, world_size):
    for k in range(3):
        if pos[k] < 0.0:
            pos[k] += world_size[k]
        elif pos[k] >= world_size[k]:
            pos[k] -= world_size[k]


# ----------------------------
# Core update function
# ----------------------------

@njit(parallel=True)
def step_boids(positions, velocities, dt, dim, sep_rad, ali_rad, coh_rad, max_force,  sep_weight, 
            ali_weight, coh_weight, max_speed, world_size):
    n = positions.shape[0]
    accelerations = np.zeros_like(velocities)

    for i in prange(n):
        sep = np.zeros(dim)
        ali = np.zeros(dim)
        coh = np.zeros(dim)

        sep_count = 0
        ali_count = 0
        coh_count = 0

        pi = positions[i]
        vi = velocities[i]

        for j in range(n):
            if i == j:
                continue

            pj = positions[j]

            dx = pj[0] - pi[0]
            dy = pj[1] - pi[1]
            dz = pj[2] - pi[2]
            if dx > world_size[0] * 0.5:
                dx -= world_size[0]
            elif dx < -world_size[0] * 0.5:
                dx += world_size[0]

            if dy > world_size[1] * 0.5:
                dy -= world_size[1]
            elif dy < -world_size[1] * 0.5:
                dy += world_size[1]

            if dz > world_size[2] * 0.5:
                dz -= world_size[2]
            elif dz < -world_size[2] * 0.5:
                dz += world_size[2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist == 0.0:
                continue

            if dist < sep_rad:
                sep[0] -= dx / dist
                sep[1] -= dy / dist
                sep[2] -= dz / dist
                sep_count += 1

            if dist < ali_rad:
                ali += velocities[j]
                ali_count += 1

            if dist < coh_rad:
                coh += pj
                coh_count += 1

        if sep_count > 0:
            sep /= sep_count
            limit_magnitude(sep, max_force)

        if ali_count > 0:
            ali /= ali_count
            ali -= vi
            limit_magnitude(ali, max_force)

        if coh_count > 0:
            coh /= coh_count
            coh -= pi
            limit_magnitude(coh, max_force)

        accelerations[i] = (
            sep_weight * sep +
            ali_weight * ali +
            coh_weight * coh
        )

    # Integrate
    for i in prange(n):
        velocities[i] += accelerations[i]
        limit_magnitude(velocities[i], max_speed)
        positions[i] += velocities[i] * dt
        wrap_around(positions[i], world_size)



# Initialization


def create_boids(n, dim, world_size):
    positions = np.random.rand(n, dim) * world_size

    # Added this so its more evenly spread out
    phi = np.random.rand(n) * 2.0 * np.pi
    costheta = np.random.rand(n) * 2.0 - 1.0
    sintheta = np.sqrt(1.0 - costheta * costheta)

    velocities = np.column_stack((
        sintheta * np.cos(phi),
        sintheta * np.sin(phi),
        costheta
    ))

    return positions, velocities
