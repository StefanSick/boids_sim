import numpy as np
from vispy import app, scene
from package.boids_2 import create_boids, step_boids

class BoidsVisualizer(scene.SceneCanvas):
    def __init__(self, n_boids, world_size, physics_config):
       
        scene.SceneCanvas.__init__(self, keys='interactive', show=True, title="Grid-Optimized Boids")
        self.unfreeze()
        
        self.n = n_boids
        self.world_size = np.array(world_size, dtype=np.float64)
        
        # Calculate Grid Dimensions for Spatial Partitioning
        max_rad = max(physics_config['ali_rad'], physics_config['coh_rad'])
        self.grid_dims = np.array([
            int(self.world_size[0] / max_rad),
            int(self.world_size[1] / max_rad),
            int(self.world_size[2] / max_rad)
        ], dtype=np.int32)

      
        self.params = physics_config
        self.positions, self.velocities = create_boids(self.n, self.world_size)
        
        self.frame_count = 0
        # Initialize current_colors 
        self.current_colors = np.ones((self.n, 4), dtype=np.float32) 

        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.center = (0, 0, 0)
        self.view.camera.distance = self.world_size[0] * 1.5

        # Markers for Boids
        self.scatter = scene.visuals.Markers(
            pos=self.positions, 
            size=1.5,            
            edge_color=None,
            face_color=self.current_colors, 
            symbol='disc',       
            scaling=True,
            parent=self.view.scene
        )
        
        # Bounding Box for reference
        self.box = scene.visuals.Box(
            width=self.world_size[0], height=self.world_size[1], depth=self.world_size[2],
            color=(1,1,1,0.05), edge_color='white', parent=self.view.scene
        )

        # Start the update timer
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

    def on_timer(self, event):
      
        step_boids(
            self.positions,              
            self.velocities,             
            self.params['dt'],           
            self.params['sep_rad'],      
            self.params['ali_rad'],      
            self.params['coh_rad'],      
            self.params['max_force'],    
            self.params['sep_weight'],   
            self.params['ali_weight'],   
            self.params['coh_weight'],   
            self.params['max_speed'],    
            self.world_size,             
            self.grid_dims,              
            self.params['margin'],       
            self.params['turn_factor']   
        )
        
        self.frame_count += 1
        
        #Speed Coloring (Every 4 frames)
        if self.frame_count % 4 == 0:
            speeds = np.linalg.norm(self.velocities, axis=1)
            t = np.clip(speeds / self.params['max_speed'], 0.0, 1.0)
            
            # SETTINGS
            threshold = 0.4  
            transition_width = 0.1 
            
           
            low = threshold - (transition_width / 2)
            high = threshold + (transition_width / 2)
            t_sharp = np.clip((t - low) / (high - low), 0.0, 1.0)
            
            # Colors
            color_red = np.array([0.8, 0.0, 0.0, 1.0], dtype=np.float32) 
            color_yellow = np.array([1.0, 0.9, 0.2, 1.0], dtype=np.float32)
            
            # Linear blend between Red and Yellow 
            colors = color_red + (color_yellow - color_red) * t_sharp[:, np.newaxis]
            
            self.current_colors = colors.astype(np.float32)
        # Push to GPU & Redraw
        self.scatter.set_data(
            pos=self.positions,
            face_color=self.current_colors
        )
        
        self.update()