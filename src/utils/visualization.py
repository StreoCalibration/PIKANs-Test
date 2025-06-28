import matplotlib.pyplot as plt
import numpy as np
import torch

def save_height_map(height_map_tensor: torch.Tensor, file_path: str, cmap: str = 'viridis', title: str = 'Predicted Height Map'):
    """
    Saves a height map tensor as an image file with a color bar.
    As per Logical View, this is part of the post-processing/visualization module.

    Args:
        height_map_tensor (torch.Tensor): The 2D height map tensor.
        file_path (str): The path to save the image file.
        cmap (str, optional): The colormap to use. Defaults to 'viridis'.
        title (str, optional): The title for the plot. Defaults to 'Predicted Height Map'.
    """
    print(f"Saving height map to {file_path}...")
    # Ensure tensor is on CPU and converted to numpy
    if height_map_tensor.requires_grad:
        height_map_numpy = height_map_tensor.cpu().detach().numpy()
    else:
        height_map_numpy = height_map_tensor.cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(height_map_numpy, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Height (nm)') # Add a label for the color bar

    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close(fig) # Prevent displaying the plot in interactive environments
    print(f"Visualization saved successfully.")

def save_comparison_map(predicted_map: torch.Tensor, ground_truth_map: np.ndarray, file_path: str, cmap: str = 'viridis', main_title: str = 'Height Map Comparison'):
    """
    Saves a side-by-side comparison of the predicted height map and the ground truth.
    This function helps in visually assessing the model's performance as requested in the Scenario View.

    Args:
        predicted_map (torch.Tensor): The 2D predicted height map tensor.
        ground_truth_map (np.ndarray): The 2D ground truth height map numpy array.
        file_path (str): The path to save the image file.
        cmap (str, optional): The colormap to use. Defaults to 'viridis'.
        main_title (str, optional): The main title for the entire plot. Defaults to 'Height Map Comparison'.
    """
    print(f"Saving comparison map to {file_path}...")
    predicted_numpy = predicted_map.cpu().detach().numpy() if predicted_map.requires_grad else predicted_map.cpu().numpy()

    # Determine shared color scale for consistent visualization
    vmin = min(np.min(predicted_numpy), np.min(ground_truth_map))
    vmax = max(np.max(predicted_numpy), np.max(ground_truth_map))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(main_title, fontsize=16)

    # Plot Ground Truth
    im1 = axes[0].imshow(ground_truth_map, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], label='Height (nm)', fraction=0.046, pad=0.04)

    # Plot Prediction
    im2 = axes[1].imshow(predicted_numpy, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], label='Height (nm)', fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Comparison visualization saved successfully.")