"""
Tests for Boid class behavior.
"""
import pytest
from boid import Boid
from vector2d import Vector2D
import numpy as np


class TestBoid:
    """Test cases for Boid behavior and rules."""
    
    def test_initialization(self):
        """Test boid initialization."""
        boid = Boid(100, 200, 800, 600)
        assert boid.position.x == 100
        assert boid.position.y == 200
        assert boid.width == 800
        assert boid.height == 600
    
    def test_initial_velocity_random(self):
        """Test that initial velocity is randomized."""
        boid1 = Boid(100, 200, 800, 600)
        boid2 = Boid(100, 200, 800, 600)
        # Very unlikely to have identical random velocities
        assert (boid1.velocity.x != boid2.velocity.x or 
                boid1.velocity.y != boid2.velocity.y)
    
    def test_apply_force(self):
        """Test applying force to boid."""
        boid = Boid(100, 200, 800, 600)
        force = Vector2D(1, 2)
        boid.apply_force(force)
        assert boid.acceleration.x == 1.0
        assert boid.acceleration.y == 2.0
    
    def test_update_position(self):
        """Test position update."""
        boid = Boid(100, 200, 800, 600)
        # Set a known velocity that won't be limited
        boid.velocity = Vector2D(2, 1)  # Magnitude ~2.24, less than max_speed of 4.0
        boid.acceleration = Vector2D(0, 0)  # No acceleration
        # Record the position before update
        initial_x = boid.position.x
        initial_y = boid.position.y
        boid.update()
        # Position should change by velocity (2, 1)
        assert abs(boid.position.x - (initial_x + 2)) < 0.01
        assert abs(boid.position.y - (initial_y + 1)) < 0.01
    
    def test_update_velocity(self):
        """Test velocity update with acceleration."""
        boid = Boid(100, 200, 800, 600)
        boid.velocity = Vector2D(1, 1)
        boid.acceleration = Vector2D(2, 3)
        boid.max_speed = 10.0  # Set high max speed to avoid limiting
        boid.update()
        # Velocity should be limited to max_speed
        assert boid.velocity.magnitude() <= boid.max_speed
        # But direction should be roughly in the right direction
        assert boid.velocity.x > 1.0  # Should have increased
        assert boid.velocity.y > 1.0  # Should have increased
    
    def test_acceleration_resets(self):
        """Test that acceleration resets after update."""
        boid = Boid(100, 200, 800, 600)
        boid.acceleration = Vector2D(2, 3)
        boid.update()
        assert boid.acceleration.x == 0.0
        assert boid.acceleration.y == 0.0
    
    def test_speed_limiting(self):
        """Test that speed is limited to max_speed."""
        boid = Boid(100, 200, 800, 600)
        boid.max_speed = 5.0
        boid.velocity = Vector2D(10, 10)  # Magnitude > max_speed
        boid.update()
        # Velocity magnitude should be limited
        assert abs(boid.velocity.magnitude() - 5.0) < 0.01
    
    def test_wrap_edges_right(self):
        """Test wrapping at right edge."""
        boid = Boid(810, 300, 800, 600)
        boid.wrap_edges()
        assert boid.position.x == 0
    
    def test_wrap_edges_left(self):
        """Test wrapping at left edge."""
        boid = Boid(-10, 300, 800, 600)
        boid.wrap_edges()
        assert boid.position.x == 800
    
    def test_wrap_edges_top(self):
        """Test wrapping at top edge."""
        boid = Boid(400, 610, 800, 600)
        boid.wrap_edges()
        assert boid.position.y == 0
    
    def test_wrap_edges_bottom(self):
        """Test wrapping at bottom edge."""
        boid = Boid(400, -10, 800, 600)
        boid.wrap_edges()
        assert boid.position.y == 600
    
    def test_separation_empty_list(self):
        """Test separation with no neighbors."""
        boid = Boid(100, 100, 800, 600)
        steering = boid.separation([])
        assert steering.x == 0.0
        assert steering.y == 0.0
    
    def test_separation_repels_close_boids(self):
        """Test that separation creates repulsion from close boids."""
        boid1 = Boid(100, 100, 800, 600)
        boid2 = Boid(105, 100, 800, 600)  # Very close
        
        steering = boid1.separation([boid2])
        # Should steer away from boid2 (negative x direction)
        assert steering.x < 0
    
    def test_alignment_empty_list(self):
        """Test alignment with no neighbors."""
        boid = Boid(100, 100, 800, 600)
        steering = boid.alignment([])
        assert steering.x == 0.0
        assert steering.y == 0.0
    
    def test_alignment_matches_neighbor_heading(self):
        """Test that alignment steers towards neighbor velocities."""
        boid1 = Boid(100, 100, 800, 600)
        boid1.velocity = Vector2D(0, 0)
        
        boid2 = Boid(120, 100, 800, 600)
        boid2.velocity = Vector2D(3, 0)
        
        steering = boid1.alignment([boid2])
        # Should steer in positive x direction
        assert steering.x > 0
    
    def test_cohesion_empty_list(self):
        """Test cohesion with no neighbors."""
        boid = Boid(100, 100, 800, 600)
        steering = boid.cohesion([])
        assert steering.x == 0.0
        assert steering.y == 0.0
    
    def test_cohesion_steers_toward_center(self):
        """Test that cohesion steers towards center of neighbors."""
        boid1 = Boid(100, 100, 800, 600)
        boid2 = Boid(120, 100, 800, 600)
        boid3 = Boid(120, 120, 800, 600)
        
        steering = boid1.cohesion([boid2, boid3])
        # Should steer towards average position (right and up)
        assert steering.x > 0
        assert steering.y > 0
    
    def test_flock_applies_all_behaviors(self):
        """Test that flock applies all three behaviors."""
        boid1 = Boid(100, 100, 800, 600)
        boid2 = Boid(105, 100, 800, 600)
        
        initial_acc = boid1.acceleration.magnitude()
        boid1.flock([boid2])
        final_acc = boid1.acceleration.magnitude()
        
        # Acceleration should change after flocking
        assert final_acc > initial_acc
