import numpy as np
from numba import njit, prange

@njit(inline='always')
def limit_magnitude(v, max_val):
    mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag > max_val and mag > 0.0:
        inv = max_val / mag
        v[0] *= inv; v[1] *= inv; v[2] *= inv

@njit
def enforce_speed_limits(v, max_speed):
    mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    min_speed = max_speed * 0.4  
    
    if mag > max_speed:
        v[:] *= (max_speed / mag)
    elif mag < min_speed:
        if mag > 0:
            v[:] *= (min_speed / mag)
        else:

            v[0] = np.random.uniform(-1, 1)
            v[1] = np.random.uniform(-1, 1)
            v[2] = np.random.uniform(-1, 1)
            v[:] = (v / np.linalg.norm(v)) * min_speed

@njit
def get_cell_coords(pos, world_size, grid_dims):
    half = world_size / 2.0
    gx = int((pos[0] + half[0]) / world_size[0] * grid_dims[0])
    gy = int((pos[1] + half[1]) / world_size[1] * grid_dims[1])
    gz = int((pos[2] + half[2]) / world_size[2] * grid_dims[2])
    return max(0, min(gx, grid_dims[0]-1)), max(0, min(gy, grid_dims[1]-1)), max(0, min(gz, grid_dims[2]-1))

@njit
def build_grid(positions, world_size, grid_dims):
    n = positions.shape[0]
    total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
    cell_ids = np.empty(n, dtype=np.int32)
    
    for i in range(n):
        gx, gy, gz = get_cell_coords(positions[i], world_size, grid_dims)
        cell_ids[i] = gx + gy * grid_dims[0] + gz * grid_dims[0] * grid_dims[1]
        
    sorted_indices = np.argsort(cell_ids)
    cell_starts = np.full(total_cells, -1, dtype=np.int32)
    cell_ends = np.full(total_cells, -1, dtype=np.int32)
    
    if n > 0:
        curr_cell = cell_ids[sorted_indices[0]]
        cell_starts[curr_cell] = 0
        for i in range(1, n):
            c = cell_ids[sorted_indices[i]]
            if c != curr_cell:
                cell_ends[curr_cell] = i
                curr_cell = c
                cell_starts[curr_cell] = i
        cell_ends[curr_cell] = n
    return sorted_indices, cell_starts, cell_ends

@njit
def avoid_boundaries(pos, vel, world_size, margin, turn_factor):
    half = world_size / 2.0
    
    for k in range(3):
        if pos[k] < -half[k] + margin:
            dist_pct = ((-half[k] + margin) - pos[k]) / margin
            # Apply turn factor 
            vel[k] += turn_factor * (dist_pct ** 2)
           
            if vel[k] < 0: vel[k] *= 0.9
            
        elif pos[k] > half[k] - margin:
            dist_pct = (pos[k] - (half[k] - margin)) / margin
            vel[k] -= turn_factor * (dist_pct ** 2)
            if vel[k] > 0: vel[k] *= 0.9

        #  If they a touch the wall, flip velocity
        if pos[k] < -half[k]:
            pos[k] = -half[k]
            vel[k] = abs(vel[k]) * 0.5 # Force velocity to point inward
        elif pos[k] > half[k]:
            pos[k] = half[k]
            vel[k] = -abs(vel[k]) * 0.5

@njit(parallel=True, fastmath=True)
def step_boids(positions, velocities, dt, sep_rad, ali_rad, coh_rad, max_force, 
               sep_weight, ali_weight, coh_weight, max_speed, world_size, 
               grid_dims, margin, turn_factor):
    
    n = positions.shape[0]
    accel = np.zeros_like(velocities)
    sorted_idx, c_starts, c_ends = build_grid(positions, world_size, grid_dims)
    
    sep_sq = sep_rad**2
    ali_sq = ali_rad**2
    coh_sq = coh_rad**2

    for i in prange(n):
        pi = positions[i]; vi = velocities[i]
        sep = np.zeros(3); ali = np.zeros(3); coh = np.zeros(3)
        sc = ac = cc = 0
        
        gx, gy, gz = get_cell_coords(pi, world_size, grid_dims)
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    nx, ny, nz = gx + dx, gy + dy, gz + dz
                    if nx<0 or nx>=grid_dims[0] or ny<0 or ny>=grid_dims[1] or nz<0 or nz>=grid_dims[2]:
                        continue
                    
                    cid = nx + ny*grid_dims[0] + nz*grid_dims[0]*grid_dims[1]
                    start, end = c_starts[cid], c_ends[cid]
                    if start == -1: continue
                    
                    for ptr in range(start, end):
                        j = sorted_idx[ptr]
                        if i == j: continue
                        
                        dx_v = positions[j,0]-pi[0]; dy_v = positions[j,1]-pi[1]; dz_v = positions[j,2]-pi[2]
                        d2 = dx_v*dx_v + dy_v*dy_v + dz_v*dz_v
                        
                        if d2 < sep_sq and d2 > 0:
                            dist = np.sqrt(d2)
                            sep[0] -= dx_v/dist; sep[1] -= dy_v/dist; sep[2] -= dz_v/dist
                            sc += 1
                        if d2 < ali_sq:
                            ali += velocities[j]; ac += 1
                        if d2 < coh_sq:
                            coh += positions[j]; cc += 1

        if sc > 0: sep /= sc; limit_magnitude(sep, max_force)
        if ac > 0: ali /= ac; ali -= vi; limit_magnitude(ali, max_force)
        if cc > 0: coh /= cc; coh -= pi; limit_magnitude(coh, max_force)
        
        # Central Gravity 
        grav_force = np.zeros(3)
        center_bias_strength = 0.0002  
        grav_force[0] = -pi[0] * center_bias_strength
        grav_force[1] = -pi[1] * center_bias_strength
        grav_force[2] = -pi[2] * center_bias_strength
        gravity_start = 650.0   # Distance from center where pull begins
        gravity_str = 0.2       # Pull strength
        
        for k in range(3):

            val = pi[k]
            abs_val = abs(val)
            
           
            if abs_val > gravity_start:
                grav_force[k] = -np.sign(val) * (abs_val - gravity_start) * gravity_str
      
        wander = (np.random.randn(3)) * 0.3
        accel[i] = (sep_weight * sep) + (ali_weight * ali) + (coh_weight * coh) + grav_force + wander

    # Final integration
    for i in prange(n):
        velocities[i] += accel[i]
        avoid_boundaries(positions[i], velocities[i], world_size, margin, turn_factor)
        enforce_speed_limits(velocities[i], max_speed)
        positions[i] += velocities[i] * dt

def create_boids(n, world_size):
    half = world_size / 2.0
    pos = (np.random.rand(n, 3) * world_size) - half
    vel = np.random.randn(n, 3) * 10.0
    return pos, vel