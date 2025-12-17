#!/usr/bin/env python3
"""
Simple demo script to run the boids simulation.
Run with: python3 demo.py
"""
from simulation import BoidsSimulation
from visualization import BoidsVisualization


def main():
    print("=" * 70)
    print("BOIDS SIMULATION DEMO")
    print("=" * 70)
    print()
    print("Creating simulation with 200 boids...")
    
    # Create simulation
    sim = BoidsSimulation(width=800, height=600, num_boids=200)
    
    print(f"✓ Created {sim.num_boids} boids")
    print(f"✓ Simulation space: {sim.width}x{sim.height}")
    print(f"✓ Using spatial grid with cell size: {sim.spatial_grid.cell_size}")
    print()
    
    # Run a few frames to warm up
    print("Running warm-up frames...")
    for _ in range(10):
        sim.update()
    
    print(f"✓ Average FPS: {sim.get_average_fps():.1f}")
    print(f"✓ Average frame time: {sim.get_average_frame_time():.2f}ms")
    print()
    
    # Reset and run more frames
    sim.reset_performance_stats()
    print("Running 100 frames...")
    for i in range(100):
        sim.update()
        if (i + 1) % 25 == 0:
            print(f"  Frame {i+1}/100 - FPS: {sim.get_average_fps():.1f}")
    
    print()
    print("=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    print(f"Average FPS: {sim.get_average_fps():.1f}")
    print(f"Average frame time: {sim.get_average_frame_time():.2f}ms")
    print()
    print("To see the visualization, run: python3 visualization.py")
    print("(Note: Requires display/GUI environment)")
    print()


if __name__ == '__main__':
    main()
