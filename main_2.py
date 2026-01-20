import sys
from vispy import app
from package.vis_2 import BoidsVisualizer

if __name__ == "__main__":
    NUM_BOIDS = 15000
    WORLD_DIMS = (1500, 1500, 1500) 

    PARAMS = {
    'dt': 0.05,
    'sep_rad': 20.0,    
    'ali_rad': 50.0,    
    'coh_rad': 20.0,    
    'max_force': 5.0,   
    'max_speed': 70,
    
    'sep_weight': 55.5,  
    'ali_weight': 2.0,  
    'coh_weight': 0.05,  
    
    'margin': 10.0,     
    'turn_factor': 1.5  
    }
    viz = BoidsVisualizer(n_boids=NUM_BOIDS, world_size=WORLD_DIMS, physics_config=PARAMS)
    if sys.flags.interactive != 1:
        app.run()