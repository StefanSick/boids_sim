"""
Spatial grid for efficient neighbor queries in boids simulation.
Uses a hash-based grid to partition space and achieve O(n) neighbor lookups.
"""
from collections import defaultdict
import math


class SpatialGrid:
    """
    A spatial hash grid for efficient neighbor queries.
    Divides space into cells and allows O(1) insertion and O(1) average neighbor queries.
    """
    
    def __init__(self, cell_size):
        """
        Initialize spatial grid.
        
        Args:
            cell_size: Size of each grid cell (should be >= perception radius)
        """
        self.cell_size = cell_size
        self.grid = defaultdict(list)
    
    def clear(self):
        """Clear all boids from the grid."""
        self.grid.clear()
    
    def _get_cell_coords(self, x, y):
        """Convert world coordinates to grid cell coordinates."""
        return (
            math.floor(x / self.cell_size),
            math.floor(y / self.cell_size)
        )
    
    def insert(self, boid):
        """Insert a boid into the grid."""
        cell = self._get_cell_coords(boid.position.x, boid.position.y)
        self.grid[cell].append(boid)
    
    def get_nearby_boids(self, boid, radius):
        """
        Get all boids within radius of the given boid.
        Only checks cells within range, significantly reducing comparisons.
        
        Args:
            boid: The boid to find neighbors for
            radius: Search radius
            
        Returns:
            List of nearby boids (excluding the boid itself)
        """
        nearby = []
        cell_x, cell_y = self._get_cell_coords(boid.position.x, boid.position.y)
        
        # Calculate how many cells to check in each direction
        cells_to_check = math.ceil(radius / self.cell_size)
        
        # Check neighboring cells
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                cell = (cell_x + dx, cell_y + dy)
                if cell in self.grid:
                    for other_boid in self.grid[cell]:
                        if other_boid is not boid:
                            # Use squared distance for efficiency
                            if boid.position.distance_squared_to(other_boid.position) <= radius * radius:
                                nearby.append(other_boid)
        
        return nearby
