"""
Tests for SpatialGrid class.
"""
import pytest
from spatial_grid import SpatialGrid
from boid import Boid


class TestSpatialGrid:
    """Test cases for SpatialGrid spatial partitioning."""
    
    def test_initialization(self):
        """Test grid initialization."""
        grid = SpatialGrid(cell_size=50)
        assert grid.cell_size == 50
        assert len(grid.grid) == 0
    
    def test_insert_single_boid(self):
        """Test inserting a single boid."""
        grid = SpatialGrid(cell_size=50)
        boid = Boid(25, 25, 800, 600)
        grid.insert(boid)
        # Grid should have one cell with one boid
        assert len(grid.grid) == 1
    
    def test_insert_multiple_boids_same_cell(self):
        """Test inserting multiple boids in the same cell."""
        grid = SpatialGrid(cell_size=50)
        boid1 = Boid(25, 25, 800, 600)
        boid2 = Boid(30, 30, 800, 600)
        grid.insert(boid1)
        grid.insert(boid2)
        # Should be in same cell (0, 0)
        assert len(grid.grid) == 1
        cell = (0, 0)
        assert len(grid.grid[cell]) == 2
    
    def test_insert_multiple_boids_different_cells(self):
        """Test inserting boids in different cells."""
        grid = SpatialGrid(cell_size=50)
        boid1 = Boid(25, 25, 800, 600)
        boid2 = Boid(75, 75, 800, 600)
        grid.insert(boid1)
        grid.insert(boid2)
        # Should be in different cells
        assert len(grid.grid) == 2
    
    def test_clear(self):
        """Test clearing the grid."""
        grid = SpatialGrid(cell_size=50)
        boid = Boid(25, 25, 800, 600)
        grid.insert(boid)
        grid.clear()
        assert len(grid.grid) == 0
    
    def test_get_nearby_boids_empty(self):
        """Test getting nearby boids from empty grid."""
        grid = SpatialGrid(cell_size=50)
        boid = Boid(25, 25, 800, 600)
        nearby = grid.get_nearby_boids(boid, radius=50)
        assert len(nearby) == 0
    
    def test_get_nearby_boids_excludes_self(self):
        """Test that nearby boids excludes the query boid itself."""
        grid = SpatialGrid(cell_size=50)
        boid = Boid(25, 25, 800, 600)
        grid.insert(boid)
        nearby = grid.get_nearby_boids(boid, radius=50)
        assert len(nearby) == 0
    
    def test_get_nearby_boids_within_radius(self):
        """Test finding boids within radius."""
        grid = SpatialGrid(cell_size=50)
        boid1 = Boid(100, 100, 800, 600)
        boid2 = Boid(110, 100, 800, 600)  # 10 units away
        boid3 = Boid(100, 110, 800, 600)  # 10 units away
        
        grid.insert(boid1)
        grid.insert(boid2)
        grid.insert(boid3)
        
        nearby = grid.get_nearby_boids(boid1, radius=20)
        assert len(nearby) == 2
        assert boid2 in nearby
        assert boid3 in nearby
    
    def test_get_nearby_boids_outside_radius(self):
        """Test that boids outside radius are not included."""
        grid = SpatialGrid(cell_size=50)
        boid1 = Boid(100, 100, 800, 600)
        boid2 = Boid(200, 200, 800, 600)  # Far away
        
        grid.insert(boid1)
        grid.insert(boid2)
        
        nearby = grid.get_nearby_boids(boid1, radius=50)
        assert len(nearby) == 0
    
    def test_get_nearby_boids_across_cells(self):
        """Test finding boids across multiple cells."""
        grid = SpatialGrid(cell_size=50)
        boid1 = Boid(49, 49, 800, 600)  # Near cell boundary
        boid2 = Boid(51, 51, 800, 600)  # In adjacent cell, close to boid1
        
        grid.insert(boid1)
        grid.insert(boid2)
        
        nearby = grid.get_nearby_boids(boid1, radius=10)
        assert len(nearby) == 1
        assert boid2 in nearby
    
    def test_efficiency_large_number_of_boids(self):
        """Test that spatial grid handles large numbers efficiently."""
        import time
        grid = SpatialGrid(cell_size=50)
        
        # Create 1000 boids
        boids = []
        import numpy as np
        for i in range(1000):
            boid = Boid(
                np.random.uniform(0, 800),
                np.random.uniform(0, 600),
                800, 600
            )
            boids.append(boid)
            grid.insert(boid)
        
        # Query should be fast (much faster than O(n²))
        start = time.time()
        for boid in boids[:10]:  # Test 10 queries
            nearby = grid.get_nearby_boids(boid, radius=50)
        elapsed = time.time() - start
        
        # Should complete quickly (< 0.1 seconds for 10 queries)
        assert elapsed < 0.1
