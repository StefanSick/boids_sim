import numpy as np
from vispy import app, scene
from vispy.color import Color
from package.boids_2 import create_boids, step_boids


class BoidsVisualizer(scene.SceneCanvas):
    def __init__(self, n_boids, world_size, physics_config):
        scene.SceneCanvas.__init__(self, keys='interactive', show=True, title="Grid-Optimized Boids")
        self.unfreeze()
        self.bgcolor = (0.88, 0.88, 0.88, 1.0)  # light gray

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
        
        self._colors = np.empty((self.n, 4), dtype=np.float32)
        self._frame = 0
        self._color_every = 2  # update colors every N frames

        # Optional: tweakable "light direction" for fake shading
        self._light_dir = np.array([0.3, 0.6, 0.7], dtype=np.float64)
        self._light_dir /= (np.linalg.norm(self._light_dir) + 1e-8)

        # Initialize colors once
        self._update_colors()

        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.center = (0, 0, 0)
        self.view.camera.distance = self.world_size[0] * 1.5

        # Inside BoidsVisualizer.__init__
        self.scatter = scene.visuals.Markers(
            pos=self.positions, 
            size=1.0,            # This is now in WORLD units (2.0 out of 900)
            edge_color=None,     # Removes the border for a cleaner "dot" look
            face_color=self._colors, 
            symbol='arrow',       # Options: 'disc', 'arrow', 'ring', 'clover', etc.
            scaling=True,        # <--- THIS IS THE KEY CHANGE
            parent=self.view.scene
        )

        self.box = scene.visuals.Box(
            width=self.world_size[0],
            height=self.world_size[1],
            depth=self.world_size[2],
            color=(1.0, 1.0, 1.0, 0.18),      # white interior fill (slightly transparent)
            edge_color=(0.0, 0.0, 0.0, 0.75), # black edges for contrast
            parent=self.view.scene
        )

        # Proper blending for transparency
        self.box.set_gl_state('translucent', depth_test=True)

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

    def _update_colors(self) -> None:
        """Update self._colors in-place (fast)."""
        half = self.world_size / 2.0

        # Position -> base RGB
        rgb = (self.positions + half) / self.world_size
        np.clip(rgb, 0.0, 1.0, out=rgb)

        # Velocity direction -> brightness
        v = self.velocities.astype(np.float64, copy=False)
        vnorm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vdir = v / vnorm

        brightness = (vdir @ self._light_dir) * 0.5 + 0.5  # [-1,1] -> [0,1]
        brightness = brightness.astype(np.float64, copy=False).reshape(-1, 1)

        # shaded_rgb = rgb * (0.35 + 0.65 * brightness)
        shade = (0.20 + 0.80 * brightness)
        shaded_rgb = rgb * shade
        shaded_rgb = np.power(shaded_rgb, 0.8)

        # Write into preallocated float32 RGBA buffer
        self._colors[:, 0:3] = shaded_rgb.astype(np.float32, copy=False)
        self._colors[:, 3] = 1.0        

    def on_timer(self, event):
        step_boids(
            self.positions, self.velocities, self.params['dt'],
            self.params['sep_rad'], self.params['ali_rad'], self.params['coh_rad'],
            self.params['max_force'], self.params['sep_weight'], self.params['ali_weight'],
            self.params['coh_weight'], self.params['max_speed'], self.world_size,
            self.grid_dims, self.params['margin'], self.params['turn_factor']
        )
        self._frame += 1

        if (self._frame % self._color_every) == 0:
            self._update_colors()
            self.scatter.set_data(pos=self.positions, face_color=self._colors)
        else:
            self.scatter.set_data(pos=self.positions, face_color=self._colors)

        self.update()