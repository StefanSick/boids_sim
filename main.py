import numpy as np
import sys
import vispy
from pathlib import Path
vispy.use('PyQt5')
from vispy import app

_main_dir = Path(__file__).resolve().parent
if str(_main_dir) not in sys.path:
    sys.path.insert(0, str(_main_dir))
from package.vis import BoidsVisualizer

if __name__ == "__main__":

    NUM_BOIDS = 5000

    WORLD_DIMS = (900, 900, 900) 

    PARAMS = {
        'dt':         0.1,
    
        'sep_rad':    15.0,  
        'ali_rad':    30.0, 
        'coh_rad':    30.0,  
        
        'max_force':  0.5,
        'max_speed':  40.0,

        'sep_weight': 2.0,   
        'ali_weight': 0.4,  
        'coh_weight': 0.10,  
    }

    # --- C. Run ---
    viz = BoidsVisualizer(n_boids=NUM_BOIDS, world_size=WORLD_DIMS, physics_config=PARAMS)
    
    if sys.flags.interactive != 1:
        app.run()   
