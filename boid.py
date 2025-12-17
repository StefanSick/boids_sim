"""
Boid class representing an individual agent in the flock.
Implements the three core rules: separation, alignment, and cohesion.
"""
from vector2d import Vector2D
import numpy as np


class Boid:
    """
    A boid (bird-oid object) that follows flocking rules.
    """
    
    def __init__(self, x, y, width, height):
        """
        Initialize a boid.
        
        Args:
            x, y: Initial position
            width, height: Boundaries for wrapping
        """
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1)
        )
        self.acceleration = Vector2D(0, 0)
        
        # Boundary settings
        self.width = width
        self.height = height
        
        # Behavior parameters (tunable for different behaviors)
        self.max_speed = 4.0
        self.max_force = 0.1
        
        # Perception radii
        self.perception_radius = 50.0
        self.separation_radius = 25.0
    
    def apply_force(self, force):
        """Apply a force to the boid's acceleration."""
        self.acceleration = self.acceleration + force
    
    def separation(self, boids):
        """
        Separation: Steer to avoid crowding local flockmates.
        Efficient implementation using squared distances.
        """
        steering = Vector2D(0, 0)
        total = 0
        
        for other in boids:
            dist_sq = self.position.distance_squared_to(other.position)
            
            # Only consider boids within separation radius
            if dist_sq < self.separation_radius * self.separation_radius and dist_sq > 0:
                # Calculate difference vector
                diff = self.position - other.position
                # Weight by distance (closer = stronger repulsion)
                diff = diff / dist_sq  # Using squared distance for efficiency
                steering = steering + diff
                total += 1
        
        if total > 0:
            steering = steering / total
            # Implement Reynolds' steering formula
            steering = steering.normalize() * self.max_speed
            steering = steering - self.velocity
            steering = steering.limit(self.max_force)
        
        return steering
    
    def alignment(self, boids):
        """
        Alignment: Steer towards the average heading of local flockmates.
        """
        steering = Vector2D(0, 0)
        total = 0
        
        for other in boids:
            dist_sq = self.position.distance_squared_to(other.position)
            
            if dist_sq < self.perception_radius * self.perception_radius:
                steering = steering + other.velocity
                total += 1
        
        if total > 0:
            steering = steering / total
            steering = steering.normalize() * self.max_speed
            steering = steering - self.velocity
            steering = steering.limit(self.max_force)
        
        return steering
    
    def cohesion(self, boids):
        """
        Cohesion: Steer towards the average position of local flockmates.
        """
        steering = Vector2D(0, 0)
        total = 0
        
        for other in boids:
            dist_sq = self.position.distance_squared_to(other.position)
            
            if dist_sq < self.perception_radius * self.perception_radius:
                steering = steering + other.position
                total += 1
        
        if total > 0:
            steering = steering / total
            # Steer towards the average position
            steering = steering - self.position
            steering = steering.normalize() * self.max_speed
            steering = steering - self.velocity
            steering = steering.limit(self.max_force)
        
        return steering
    
    def flock(self, boids):
        """
        Apply all three flocking behaviors.
        Uses precomputed neighbor list for efficiency.
        """
        separation = self.separation(boids)
        alignment = self.alignment(boids)
        cohesion = self.cohesion(boids)
        
        # Weight the different behaviors
        separation = separation * 1.5
        alignment = alignment * 1.0
        cohesion = cohesion * 1.0
        
        self.apply_force(separation)
        self.apply_force(alignment)
        self.apply_force(cohesion)
    
    def update(self):
        """Update boid position and velocity."""
        # Update velocity
        self.velocity = self.velocity + self.acceleration
        # Limit speed
        self.velocity = self.velocity.limit(self.max_speed)
        # Update position
        self.position = self.position + self.velocity
        # Reset acceleration
        self.acceleration = Vector2D(0, 0)
    
    def wrap_edges(self):
        """Wrap boid around screen edges."""
        if self.position.x > self.width:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = self.width
        
        if self.position.y > self.height:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = self.height
