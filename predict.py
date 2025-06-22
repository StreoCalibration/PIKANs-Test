import argparse
import yaml
import torch
import os

from src.model.pikans import PIKANs
from src.data_loader.datasets import load_inference_data
from src.utils.visualization import save_height_map

def predict(model_path, data_path, output_dir, config_path):
    """
    Performs inference using a trained PIKANs model.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration to build model architecture
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
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
    save_height_map(predicted_height_map, output_path)

    print("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict height map using a trained PIKANs model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file (.pth).")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input data directory (e.g., 'data/raw/benchmark/sample_001').")
    parser.add_argument('--output_dir', type=str, default="outputs/predictions", help="Directory to save the output.")
    parser.add_argument('--config', type=str, default="configs/default_config.yaml", help="Path to the configuration file used for training.")
    args = parser.parse_args()
    predict(args.model_path, args.data_path, args.output_dir, args.config)