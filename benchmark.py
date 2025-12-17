"""
Benchmark script to measure and compare efficiency of the boids simulation.
Compares spatial grid approach vs naive O(n²) approach.
"""
import time
import numpy as np
from simulation import BoidsSimulation
from boid import Boid


class NaiveBoidsSimulation(BoidsSimulation):
    """
    Naive O(n²) implementation for comparison.
    Checks every boid against every other boid.
    """
    
    def update(self):
        """Update using naive O(n²) approach."""
        start_time = time.time()
        
        # Update each boid by checking ALL other boids
        for boid in self.boids:
            # Naive: check all other boids (O(n²))
            other_boids = [b for b in self.boids if b is not boid]
            boid.flock(other_boids)
        
        # Update positions
        for boid in self.boids:
            boid.update()
            boid.wrap_edges()
        
        # Track performance
        elapsed = time.time() - start_time
        self.total_time += elapsed
        self.frame_count += 1


def benchmark_simulation(sim_class, num_boids, num_frames=50):
    """
    Benchmark a simulation.
    
    Args:
        sim_class: Simulation class to test
        num_boids: Number of boids
        num_frames: Number of frames to simulate
    
    Returns:
        Average time per frame in milliseconds
    """
    sim = sim_class(width=800, height=600, num_boids=num_boids)
    
    # Warm up
    for _ in range(5):
        sim.update()
    
    # Reset and measure
    sim.reset_performance_stats()
    start_time = time.time()
    
    for _ in range(num_frames):
        sim.update()
    
    elapsed = time.time() - start_time
    return (elapsed / num_frames) * 1000  # ms per frame


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 70)
    print("BOIDS SIMULATION EFFICIENCY BENCHMARK")
    print("=" * 70)
    print()
    
    boid_counts = [50, 100, 200, 400]
    
    print(f"{'Boids':<10} {'Spatial Grid (ms)':<20} {'Naive O(n²) (ms)':<20} {'Speedup':<10}")
    print("-" * 70)
    
    results = []
    
    for num_boids in boid_counts:
        # Benchmark efficient version with spatial grid
        time_efficient = benchmark_simulation(BoidsSimulation, num_boids, num_frames=30)
        
        # Benchmark naive version (limit frames for larger counts)
        frames = 30 if num_boids <= 200 else 10
        time_naive = benchmark_simulation(NaiveBoidsSimulation, num_boids, num_frames=frames)
        
        speedup = time_naive / time_efficient
        results.append((num_boids, time_efficient, time_naive, speedup))
        
        print(f"{num_boids:<10} {time_efficient:<20.2f} {time_naive:<20.2f} {speedup:<10.1f}x")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The spatial grid approach provides significant performance improvements:")
    print()
    
    for num_boids, time_eff, time_naive, speedup in results:
        fps_eff = 1000 / time_eff
        fps_naive = 1000 / time_naive
        print(f"  {num_boids} boids:")
        print(f"    - Spatial Grid: {fps_eff:.1f} FPS ({time_eff:.2f}ms per frame)")
        print(f"    - Naive O(n²):  {fps_naive:.1f} FPS ({time_naive:.2f}ms per frame)")
        print(f"    - Speedup: {speedup:.1f}x faster")
        print()
    
    print("Key Insights:")
    print("  - Spatial grid achieves O(n) complexity vs O(n²) for naive approach")
    print("  - Performance improvement scales with number of boids")
    print("  - Spatial partitioning enables real-time simulation with many agents")
    print()


if __name__ == '__main__':
    run_benchmarks()
