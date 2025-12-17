import sys
import numpy as np
from vispy import app, scene
from numba import njit, prange
from package.boids import create_boids, step_boids

class BoidsVisualizer(scene.SceneCanvas):
    def __init__(self, n_boids, world_size, physics_config):
        # Initialize Canvas
        scene.SceneCanvas.__init__(self, keys='interactive', show=True, title="Numba Boids 3D")
        self.unfreeze()
        
        # Sim State
        self.n = n_boids
        self.dim = 3
      
        self.world_size = np.array(world_size, dtype=np.float64) 
        
     
        self.dt         = physics_config.get('dt', 0.5)
        self.sep_rad    = physics_config.get('sep_rad', 4.0)
        self.ali_rad    = physics_config.get('ali_rad', 7.0)
        self.coh_rad    = physics_config.get('coh_rad', 7.0)
        self.max_force  = physics_config.get('max_force', 0.5)
        self.sep_weight = physics_config.get('sep_weight', 1.5)
        self.ali_weight = physics_config.get('ali_weight', 1.0)
        self.coh_weight = physics_config.get('coh_weight', 1.0)
        self.max_speed  = physics_config.get('max_speed', 2.0)

        # Initialize Data 
       
        self.positions, self.velocities = create_boids(self.n, self.dim, self.world_size)
        
        # Visualization
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = self.world_size[0] * 1.5
      
        self.view.camera.center = (self.world_size[0]/2, self.world_size[1]/2, self.world_size[2]/2)

        # Boid Markers
        self.scatter = scene.visuals.Markers(
            pos=self.positions, 
            size=0.01, 
            edge_color=None,
            face_color='white', 
            parent=self.view.scene, 
            scaling=True
        )

        # Boundary Box
        self.box = scene.visuals.Box(
            width=self.world_size[0], 
            height=self.world_size[1], 
            depth=self.world_size[2],
            color=(1, 1, 1, 0.1), 
            edge_color='white',
            parent=self.view.scene
        )
        self.box.transform = scene.transforms.STTransform(
            translate=(self.world_size[0]/2, self.world_size[1]/2, self.world_size[2]/2)
        )

        # Timer
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

    def on_timer(self, event):
       
        step_boids(
            self.positions, 
            self.velocities, 
            self.dt, 
            self.dim,
            self.sep_rad, 
            self.ali_rad, 
            self.coh_rad, 
            self.max_force, 
            self.sep_weight, 
            self.ali_weight, 
            self.coh_weight, 
            self.max_speed, 
            self.world_size
        )
        self.scatter.set_data(pos=self.positions)
        self.update()