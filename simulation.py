"""
Main simulation class for the boids flocking behavior.
Uses spatial partitioning for efficient neighbor queries.
"""
from boid import Boid
from spatial_grid import SpatialGrid
import time


class BoidsSimulation:
    """
    Main simulation class that manages all boids and their interactions.
    Uses spatial grid for O(n) performance instead of O(n²).
    """
    
    def __init__(self, width=800, height=600, num_boids=100):
        """
        Initialize the simulation.
        
        Args:
            width: Width of simulation space
            height: Height of simulation space
            num_boids: Number of boids to create
        """
        self.width = width
        self.height = height
        self.num_boids = num_boids
        
        # Create boids
        self.boids = []
        import numpy as np
        for _ in range(num_boids):
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            self.boids.append(Boid(x, y, width, height))
        
        # Create spatial grid for efficient neighbor queries
        # Cell size should be at least as large as perception radius
        self.spatial_grid = SpatialGrid(cell_size=50.0)
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0.0
    
    def update(self):
        """
        Update simulation by one step.
        Uses spatial grid to achieve O(n) complexity.
        """
        start_time = time.time()
        
        # Clear and rebuild spatial grid
        self.spatial_grid.clear()
        for boid in self.boids:
            self.spatial_grid.insert(boid)
        
        # Update each boid using nearby boids only
        for boid in self.boids:
            # Get nearby boids efficiently using spatial grid
            nearby = self.spatial_grid.get_nearby_boids(boid, boid.perception_radius)
            boid.flock(nearby)
        
        # Update positions
        for boid in self.boids:
            boid.update()
            boid.wrap_edges()
        
        # Track performance
        elapsed = time.time() - start_time
        self.total_time += elapsed
        self.frame_count += 1
    
    def get_average_fps(self):
        """Get average frames per second."""
        if self.total_time > 0:
            return self.frame_count / self.total_time
        return 0.0
    
    def get_average_frame_time(self):
        """Get average time per frame in milliseconds."""
        if self.frame_count > 0:
            return (self.total_time / self.frame_count) * 1000
        return 0.0
    
    def reset_performance_stats(self):
        """Reset performance tracking."""
        self.frame_count = 0
        self.total_time = 0.0
