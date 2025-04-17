import numpy as np

def estimate_speed(vehicle):
    # Assuming we have a constant distance between frames
    distance_per_pixel = 0.05  # meters
    frame_rate = 30  # frames per second

    # Mock speed calculation based on bounding box movement
    speed = np.random.uniform(10, 60)  # Random speed for example
    return speed
