import argparse
import yaml
import torch
import os
import numpy as np

from src.model.pikans import PIKANs
from src.data_loader.datasets import load_inference_data
from src.utils.visualization import save_height_map, save_comparison_map
from src.preprocessing.filters import select_roi

def predict(model_path, data_path, output_dir, config_path):
    """
    Performs inference using a trained PIKANs model.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration to build model architecture
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_conf = config['model']

    print(f"Loading model from {model_path}")
    # Initialize model with the same architecture as during training
    model = PIKANs(
        layer_widths=model_conf['kan_layers'],
        grid_size=model_conf['grid_size'],
        spline_order=model_conf['spline_order']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loading and preprocessing data from {data_path}")
    data_conf = config['data']
    input_tensor, original_shape = load_inference_data(
        data_path,
        data_layout=config['data']['layout'],
        normalization_method=data_conf['normalization'],
        roi=data_conf.get('roi')
    )
    input_tensor = input_tensor.to(device)

    print("Running inference...")
    with torch.no_grad():
        predicted_height_vector = model(input_tensor)

    # Post-processing and saving
    predicted_height_map = predicted_height_vector.reshape(original_shape)
    sample_name = os.path.basename(os.path.normpath(data_path))
    output_filename = f"predicted_height_{sample_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    save_height_map(predicted_height_map, output_path, title=f"Predicted Height for {sample_name}")

    # Try to load Ground Truth for comparison, as per Scenario View
    gt_path = os.path.join(data_path, "gt", "height.npy")
    if os.path.exists(gt_path):
        print(f"Ground truth found at {gt_path}. Creating comparison image.")
        ground_truth_map = np.load(gt_path)

        # Apply the same ROI to the ground truth map if it was used for inference
        if data_conf.get('roi'):
            ground_truth_map = select_roi(ground_truth_map, data_conf.get('roi'))

        # Check if shapes match after potential ROI
        if ground_truth_map.shape != original_shape:
            print(f"Warning: Ground truth shape {ground_truth_map.shape} does not match prediction shape {original_shape}. Skipping comparison.")
        else:
            comparison_filename = f"comparison_{sample_name}.png"
            comparison_path = os.path.join(output_dir, comparison_filename)
            save_comparison_map(
                predicted_map=predicted_height_map,
                ground_truth_map=ground_truth_map,
                file_path=comparison_path,
                main_title=f"Prediction vs. GT for {sample_name}"
            )
    else:
        print("Ground truth not found, skipping comparison visualization.")

    print("\nInference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict height map using a trained PIKANs model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file (.pth).")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input data directory for a single sample (e.g., 'data/4w3b_benchmark/sample_001').")
    parser.add_argument('--output_dir', type=str, default="outputs/predictions", help="Directory to save the output.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file used for training the model (e.g., 'configs/4w3b_config.yaml').")
    args = parser.parse_args()
    predict(args.model_path, args.data_path, args.output_dir, args.config)