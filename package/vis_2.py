import numpy as np
from vispy import app, scene
from package.boids_2 import create_boids, step_boids

class BoidsVisualizer(scene.SceneCanvas):
    def __init__(self, n_boids, world_size, physics_config):
        scene.SceneCanvas.__init__(self, keys='interactive', show=True, title="Grid-Optimized Boids")
        self.unfreeze()
        
        self.n = n_boids
        self.world_size = np.array(world_size, dtype=np.float64)
        # Grid dimensions: usually world_size / max_radius
        max_rad = max(physics_config['ali_rad'], physics_config['coh_rad'])
        self.grid_dims = np.array([
            int(self.world_size[0] / max_rad),
            int(self.world_size[1] / max_rad),
            int(self.world_size[2] / max_rad)
        ], dtype=np.int32)

        self.params = physics_config
        self.positions, self.velocities = create_boids(self.n, self.world_size)
        
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.center = (0, 0, 0)
        self.view.camera.distance = self.world_size[0] * 1.5

        # Inside BoidsVisualizer.__init__
        self.scatter = scene.visuals.Markers(
            pos=self.positions, 
            size=1.0,            # This is now in WORLD units (2.0 out of 900)
            edge_color=None,     # Removes the border for a cleaner "dot" look
            face_color='white', 
            symbol='arrow',       # Options: 'disc', 'arrow', 'ring', 'clover', etc.
            scaling=True,        # <--- THIS IS THE KEY CHANGE
            parent=self.view.scene
        )
        self.box = scene.visuals.Box(width=self.world_size[0], height=self.world_size[1], depth=self.world_size[2],
                                     color=(1,1,1,0.05), edge_color='white', parent=self.view.scene)

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

    def on_timer(self, event):
        step_boids(
            self.positions, self.velocities, self.params['dt'],
            self.params['sep_rad'], self.params['ali_rad'], self.params['coh_rad'],
            self.params['max_force'], self.params['sep_weight'], self.params['ali_weight'],
            self.params['coh_weight'], self.params['max_speed'], self.world_size,
            self.grid_dims, self.params['margin'], self.params['turn_factor']
        )
        self.scatter.set_data(pos=self.positions)
        self.update()