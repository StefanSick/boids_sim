"""
Visualization module for the boids simulation.
Uses matplotlib for animation.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from simulation import BoidsSimulation
import numpy as np


class BoidsVisualization:
    """Visualize the boids simulation using matplotlib."""
    
    def __init__(self, simulation):
        """
        Initialize visualization.
        
        Args:
            simulation: BoidsSimulation instance
        """
        self.sim = simulation
        
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.ax.set_xlim(0, simulation.width)
        self.ax.set_ylim(0, simulation.height)
        self.ax.set_aspect('equal')
        self.ax.set_title('Boids Simulation (Efficient with Spatial Grid)')
        
        # Create scatter plot for boids
        self.scatter = self.ax.scatter([], [], s=30, c='blue', alpha=0.6)
        
        # Text for stats
        self.stats_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    def init(self):
        """Initialize animation."""
        self.scatter.set_offsets(np.empty((0, 2)))
        self.stats_text.set_text('')
        return self.scatter, self.stats_text
    
    def update_frame(self, frame):
        """Update animation frame."""
        # Update simulation
        self.sim.update()
        
        # Get boid positions
        positions = np.array([[b.position.x, b.position.y] for b in self.sim.boids])
        self.scatter.set_offsets(positions)
        
        # Update stats
        avg_fps = self.sim.get_average_fps()
        avg_frame_time = self.sim.get_average_frame_time()
        stats = f'Boids: {self.sim.num_boids}\n'
        stats += f'FPS: {avg_fps:.1f}\n'
        stats += f'Frame Time: {avg_frame_time:.2f}ms'
        self.stats_text.set_text(stats)
        
        return self.scatter, self.stats_text
    
    def animate(self, frames=500, interval=20):
        """
        Run the animation.
        
        Args:
            frames: Number of frames to animate
            interval: Delay between frames in milliseconds
        """
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, init_func=self.init,
            frames=frames, interval=interval, blit=True
        )
        plt.show()
        return anim


def main():
    """Run a simple visualization."""
    # Create simulation with 200 boids
    sim = BoidsSimulation(width=800, height=600, num_boids=200)
    
    # Create and run visualization
    viz = BoidsVisualization(sim)
    viz.animate(frames=1000, interval=20)


if __name__ == '__main__':
    main()
