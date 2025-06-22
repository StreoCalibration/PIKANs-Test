import matplotlib.pyplot as plt
import numpy as np

def save_height_map(height_map_tensor, file_path, cmap='viridis'):
    """
    Saves a height map tensor as an image file.
    As per Logical View, this is part of the post-processing/visualization module.
    """
    print(f"Saving height map to {file_path}...")
    # Ensure tensor is on CPU and converted to numpy
    height_map_numpy = height_map_tensor.cpu().detach().numpy()

    # Using matplotlib's imsave handles normalization and colormapping.
    plt.imsave(file_path, height_map_numpy, cmap=cmap)
    print(f"Visualization saved successfully.")