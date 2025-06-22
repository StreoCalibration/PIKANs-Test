import argparse
import yaml
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader.datasets import InterferometryDataset
from src.model.pikans import PIKANs
from src.loss.physics_informed_loss import PhysicsLoss

def train(config):
    """
    Main training loop for the PIKANs model.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading configuration...")
    print(config)

    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['output']['log_dir'], timestamp)
    model_dir = os.path.join(config['output']['model_dir'], timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}")
    print(f"Models will be saved to: {model_dir}")

    # 1. Data Loading (from Scenario/Logical View)
    print("\n1. Initializing Dataloader...")
    data_conf = config['data']
    train_dataset = InterferometryDataset(
        data_dir=data_conf['path'],
        gt_dir=data_conf['gt_path'],
        roi=data_conf['roi'],
        normalization_method=data_conf['normalization']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4, # Adjust based on your system
        pin_memory=True
    )

    # 2. Model Initialization (from Logical/Development View)
    print("\n2. Initializing PIKANs model...")
    model_conf = config['model']
    model = PIKANs(
        layer_widths=model_conf['kan_layers'],
        grid_size=model_conf['grid_size'],
        spline_order=model_conf['spline_order']
    ).to(device)

    # 3. Loss Function and Optimizer
    print("\n3. Setting up loss function and optimizer...")
    criterion_mse = torch.nn.MSELoss()
    criterion_physics = PhysicsLoss(
        wavelengths=torch.tensor(config['model']['wavelengths'], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Optional: Learning rate scheduler
    if config['training']['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    # 4. Training Loop (from Process View)
    print("\n4. Starting training loop...")
    loss_weights = config['training']['loss_weights']
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False)
        for batch_data in progress_bar:
            inputs, targets = batch_data
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(inputs)
            
            # Loss calculation
            mse_loss = criterion_mse(predictions, targets)
            physics_loss = criterion_physics(predictions, inputs) # Returns 0 for now
            total_loss = (mse_loss * loss_weights['mse']) + (physics_loss * loss_weights['physics'])
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            total_loss_epoch += total_loss.item()
            progress_bar.set_postfix(loss=total_loss.item())

        avg_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - Average Loss: {avg_loss:.6f}")

        if scheduler:
            scheduler.step()

    print("\nTraining finished.")

    # 5. Save Model
    final_model_path = os.path.join(model_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PIKANs model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)