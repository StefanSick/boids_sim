"""
Tests for Vector2D class.
"""
import pytest
import numpy as np
from vector2d import Vector2D


class TestVector2D:
    """Test cases for Vector2D operations."""
    
    def test_initialization(self):
        """Test vector initialization."""
        v = Vector2D(3, 4)
        assert v.x == 3.0
        assert v.y == 4.0
    
    def test_default_initialization(self):
        """Test default initialization to zero."""
        v = Vector2D()
        assert v.x == 0.0
        assert v.y == 0.0
    
    def test_addition(self):
        """Test vector addition."""
        v1 = Vector2D(1, 2)
        v2 = Vector2D(3, 4)
        v3 = v1 + v2
        assert v3.x == 4.0
        assert v3.y == 6.0
    
    def test_subtraction(self):
        """Test vector subtraction."""
        v1 = Vector2D(5, 7)
        v2 = Vector2D(2, 3)
        v3 = v1 - v2
        assert v3.x == 3.0
        assert v3.y == 4.0
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Vector2D(2, 3)
        v2 = v * 3
        assert v2.x == 6.0
        assert v2.y == 9.0
    
    def test_scalar_division(self):
        """Test scalar division."""
        v = Vector2D(6, 9)
        v2 = v / 3
        assert v2.x == 2.0
        assert v2.y == 3.0
    
    def test_division_by_zero(self):
        """Test division by zero returns zero vector."""
        v = Vector2D(6, 9)
        v2 = v / 0
        assert v2.x == 0.0
        assert v2.y == 0.0
    
    def test_magnitude(self):
        """Test magnitude calculation."""
        v = Vector2D(3, 4)
        assert v.magnitude() == 5.0
    
    def test_magnitude_squared(self):
        """Test squared magnitude calculation."""
        v = Vector2D(3, 4)
        assert v.magnitude_squared() == 25.0
    
    def test_normalize(self):
        """Test vector normalization."""
        v = Vector2D(3, 4)
        v_norm = v.normalize()
        assert abs(v_norm.magnitude() - 1.0) < 1e-10
        assert abs(v_norm.x - 0.6) < 1e-10
        assert abs(v_norm.y - 0.8) < 1e-10
    
    def test_normalize_zero_vector(self):
        """Test normalization of zero vector."""
        v = Vector2D(0, 0)
        v_norm = v.normalize()
        assert v_norm.x == 0.0
        assert v_norm.y == 0.0
    
    def test_limit(self):
        """Test limiting vector magnitude."""
        v = Vector2D(3, 4)  # magnitude = 5
        v_limited = v.limit(3)
        assert abs(v_limited.magnitude() - 3.0) < 1e-10
    
    def test_limit_no_change(self):
        """Test limit when vector is already within limit."""
        v = Vector2D(1, 1)
        v_limited = v.limit(5)
        assert v_limited.x == 1.0
        assert v_limited.y == 1.0
    
    def test_distance_to(self):
        """Test distance calculation."""
        v1 = Vector2D(0, 0)
        v2 = Vector2D(3, 4)
        assert v1.distance_to(v2) == 5.0
    
    def test_distance_squared_to(self):
        """Test squared distance calculation."""
        v1 = Vector2D(0, 0)
        v2 = Vector2D(3, 4)
        assert v1.distance_squared_to(v2) == 25.0
    
    def test_copy(self):
        """Test vector copy."""
        v1 = Vector2D(3, 4)
        v2 = v1.copy()
        assert v2.x == v1.x
        assert v2.y == v1.y
        # Ensure it's a different object
        v2.x = 5
        assert v1.x == 3.0
