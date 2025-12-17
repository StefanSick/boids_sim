"""
Efficient 2D vector operations for boids simulation.
Uses numpy for vectorized operations when possible.
"""
import numpy as np


class Vector2D:
    """Simple 2D vector class with efficient operations."""
    
    __slots__ = ['x', 'y']  # Memory optimization
    
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        """
        Divide vector by scalar.
        
        Returns zero vector on division by zero to maintain simulation stability.
        This prevents crashes when vectors have zero magnitude, which can occur
        naturally during boids simulation (e.g., when velocity is zero).
        """
        if scalar == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def magnitude(self):
        """Calculate vector magnitude efficiently."""
        return np.sqrt(self.x * self.x + self.y * self.y)
    
    def magnitude_squared(self):
        """Calculate squared magnitude (more efficient for comparisons)."""
        return self.x * self.x + self.y * self.y
    
    def normalize(self):
        """Return normalized vector."""
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector2D(0, 0)
    
    def limit(self, max_val):
        """Limit vector magnitude to max_val."""
        mag = self.magnitude()
        if mag > max_val:
            return self.normalize() * max_val
        return Vector2D(self.x, self.y)
    
    def distance_to(self, other):
        """Calculate distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)
    
    def distance_squared_to(self, other):
        """Calculate squared distance (more efficient)."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy
    
    def copy(self):
        """Return a copy of this vector."""
        return Vector2D(self.x, self.y)
    
    def __repr__(self):
        return f"Vector2D({self.x:.2f}, {self.y:.2f})"
