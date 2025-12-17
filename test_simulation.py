"""
Tests for BoidsSimulation class.
"""
import pytest
from simulation import BoidsSimulation
import time


class TestBoidsSimulation:
    """Test cases for BoidsSimulation."""
    
    def test_initialization(self):
        """Test simulation initialization."""
        sim = BoidsSimulation(width=800, height=600, num_boids=50)
        assert sim.width == 800
        assert sim.height == 600
        assert sim.num_boids == 50
        assert len(sim.boids) == 50
    
    def test_boids_created_within_bounds(self):
        """Test that all boids are created within bounds."""
        sim = BoidsSimulation(width=800, height=600, num_boids=50)
        for boid in sim.boids:
            assert 0 <= boid.position.x <= 800
            assert 0 <= boid.position.y <= 600
    
    def test_update_changes_positions(self):
        """Test that update changes boid positions."""
        sim = BoidsSimulation(width=800, height=600, num_boids=10)
        
        # Store initial positions
        initial_positions = [(b.position.x, b.position.y) for b in sim.boids]
        
        # Update simulation
        sim.update()
        
        # Check that at least some positions changed
        final_positions = [(b.position.x, b.position.y) for b in sim.boids]
        assert initial_positions != final_positions
    
    def test_spatial_grid_used(self):
        """Test that spatial grid is being used."""
        sim = BoidsSimulation(width=800, height=600, num_boids=50)
        assert sim.spatial_grid is not None
        assert sim.spatial_grid.cell_size == 50.0
    
    def test_performance_tracking(self):
        """Test performance tracking."""
        sim = BoidsSimulation(width=800, height=600, num_boids=50)
        
        # Run a few updates
        for _ in range(10):
            sim.update()
        
        # Check that performance is tracked
        assert sim.frame_count == 10
        assert sim.total_time > 0
        assert sim.get_average_fps() > 0
        assert sim.get_average_frame_time() > 0
    
    def test_reset_performance_stats(self):
        """Test resetting performance statistics."""
        sim = BoidsSimulation(width=800, height=600, num_boids=50)
        
        # Run some updates
        for _ in range(5):
            sim.update()
        
        # Reset stats
        sim.reset_performance_stats()
        
        assert sim.frame_count == 0
        assert sim.total_time == 0.0
    
    def test_efficiency_with_many_boids(self):
        """Test that simulation is efficient with many boids."""
        sim = BoidsSimulation(width=800, height=600, num_boids=200)
        
        # Run 10 frames and measure time
        start_time = time.time()
        for _ in range(10):
            sim.update()
        elapsed = time.time() - start_time
        
        # Should complete 10 frames in reasonable time (< 1 second)
        assert elapsed < 1.0
        
        # Average frame time should be reasonable
        avg_frame_time = sim.get_average_frame_time()
        assert avg_frame_time < 100  # Less than 100ms per frame
    
    def test_scalability_comparison(self):
        """Test scalability: more boids should scale reasonably."""
        # Small simulation
        sim_small = BoidsSimulation(width=800, height=600, num_boids=50)
        start = time.time()
        for _ in range(5):
            sim_small.update()
        time_small = time.time() - start
        
        # Larger simulation (4x boids)
        sim_large = BoidsSimulation(width=800, height=600, num_boids=200)
        start = time.time()
        for _ in range(5):
            sim_large.update()
        time_large = time.time() - start
        
        # With spatial grid, should be roughly linear (not quadratic)
        # 4x boids should take less than 10x time (allowing margin)
        assert time_large < time_small * 10
