import argparse
import yaml
import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the new dataset for fine-tuning
from src.data_loader.datasets import RealDataFinetuneDataset
from src.model.pikans import PIKANs
from src.loss.physics_informed_loss import PhysicsInformedLoss

def finetune(config):
    """
    Main fine-tuning loop for the PIKANs model on real data without ground truth.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading configuration for fine-tuning...")
    print(config)

    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['output']['log_dir'], f"finetune_{timestamp}")
    model_dir = os.path.join(config['output']['model_dir'], f"finetune_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}")
    print(f"Models will be saved to: {model_dir}")

    # 1. Data Loading (using the new RealDataFinetuneDataset)
    print("\n1. Initializing Dataloader for real data...")
    data_conf = config['data']
    # Use the new dataset class which does not require GT
    finetune_dataset = RealDataFinetuneDataset(
        data_dir=data_conf['path'],
        data_layout=data_conf['layout'],
        roi=data_conf.get('roi'),
        normalization_method=data_conf['normalization']
    )
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_conf.get('num_workers', 4),
        pin_memory=True
    )

    # 2. Model Initialization
    print("\n2. Initializing PIKANs model...")
    model_conf = config['model']
    model = PIKANs(
        layer_widths=model_conf['kan_layers'],
        grid_size=model_conf['grid_size'],
        spline_order=model_conf['spline_order']
    ).to(device)

    # --- Key modification: Load pre-trained model weights ---
    pretrained_path = config['training'].get('pretrained_model_path')
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading pre-trained weights from: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("\nWarning: No pre-trained model path found or file doesn't exist. Starting from scratch.")

    # 3. Loss Function and Optimizer
    print("\n3. Setting up loss function and optimizer...")
    num_buckets = config['data']['layout']['num_buckets']
    phase_shifts = torch.arange(0, num_buckets) * (2 * np.pi / num_buckets)

    # For fine-tuning, we ONLY use the physics-informed loss.
    criterion_physics = PhysicsInformedLoss(
        wavelengths=config['model']['wavelengths'],
        num_wavelengths=config['data']['layout']['num_wavelengths'],
        num_buckets=num_buckets,
        phase_shifts=phase_shifts
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    scheduler_conf = config['training'].get('scheduler')
    if scheduler_conf and scheduler_conf.get('type') == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_conf.get('step_size', 30),
            gamma=scheduler_conf.get('gamma', 0.1))
    else:
        scheduler = None

    # 4. Fine-tuning Loop
    print("\n4. Starting fine-tuning loop...")
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss_epoch = 0

        progress_bar = tqdm(finetune_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False)
        # --- Key modification: The data loader only returns inputs ---
        for inputs in progress_bar:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(inputs)

            # --- Key modification: Loss is only physics-based ---
            physics_loss = criterion_physics(predictions, inputs)
            total_loss = physics_loss # No MSE loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()
            progress_bar.set_postfix(loss=total_loss.item())

        avg_loss = total_loss_epoch / len(finetune_loader)
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - Average Loss: {avg_loss:.6f}")

        if scheduler:
            scheduler.step()

    print("\nFine-tuning finished.")

    # 5. Save Model
    final_model_path = os.path.join(model_dir, "final_bga_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PIKANs model on real data.")
    parser.add_argument('--config', type=str, required=True, help="Path to the fine-tuning configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    finetune(config)
