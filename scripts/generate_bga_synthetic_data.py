import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import math

def create_bga_height_map(
    img_shape,
    ball_diameter_um,
    ball_height_um,
    ball_pitch_um,
    pixels_per_um,
    randomize=False
):
    """
    Generates a height map for a Ball Grid Array (BGA).

    Args:
        img_shape (tuple): The (height, width) of the output image in pixels.
        ball_diameter_um (float): The diameter of the balls in micrometers.
        ball_height_um (float): The maximum height of the balls in micrometers.
        ball_pitch_um (tuple): The (y, x) spacing between ball centers in micrometers.
        pixels_per_um (float): The resolution in pixels per micrometer.
        randomize (bool): If True, adds random variations to ball properties.

    Returns:
        np.ndarray: The generated height map.
    """
    height_map = np.zeros(img_shape, dtype=np.float32)
    img_height_px, img_width_px = img_shape

    ball_radius_um = ball_diameter_um / 2.0
    ball_radius_px = ball_radius_um * pixels_per_um

    # Create coordinate grid for a single ball patch
    patch_size = int(2 * ball_radius_px)
    if patch_size % 2 == 0:
        patch_size += 1

    radius_px = patch_size // 2
    y_coords = np.arange(-radius_px, radius_px + 1)
    x_coords = np.arange(-radius_px, radius_px + 1)
    y, x = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Convert pixel coordinates to micrometers
    x_um = x / pixels_per_um
    y_um = y / pixels_per_um

    # Calculate the squared distance from the center in micrometers
    r2_um = x_um**2 + y_um**2

    # Create a single ball using the sphere equation (z = sqrt(R^2 - r^2))
    # We use ball_radius_um as the sphere's radius and calculate height
    ball_profile_um = np.sqrt(np.maximum(0, ball_radius_um**2 - r2_um))

    # Clip the height to the specified ball_height_um
    # This creates a spherical cap, not a full hemisphere if height < radius
    ball_profile_um[ball_profile_um > ball_height_um] = 0 # Should not happen if height <= radius

    # Since the problem states the ball is half-buried, the sphere's center is below the substrate.
    # The sphere radius is `R`. The height of the cap is `H`. `z = sqrt(R^2 - r^2) - (R-H)`
    # For a hemisphere, R=H. So z = sqrt(H^2-r^2). The code is correct for this case.

    # Convert final profile height to nanometers as is convention
    ball_profile_nm = ball_profile_um * 1000.0

    # Calculate ball positions in pixels
    pitch_y_px = ball_pitch_um[0] * pixels_per_um
    pitch_x_px = ball_pitch_um[1] * pixels_per_um

    num_balls_y = int(img_height_px / pitch_y_px)
    num_balls_x = int(img_width_px / pitch_x_px)

    for i in range(num_balls_y):
        for j in range(num_balls_x):
            center_y = int((i + 0.5) * pitch_y_px)
            center_x = int((j + 0.5) * pitch_x_px)

            current_ball_profile = ball_profile_nm.copy()

            # --- Domain Randomization ---
            if randomize:
                # Randomize height (±10%)
                height_factor = 1.0 + np.random.uniform(-0.1, 0.1)
                current_ball_profile *= height_factor

                # Randomize position (±5% of pitch)
                dx = int(np.random.uniform(-0.05, 0.05) * pitch_x_px)
                dy = int(np.random.uniform(-0.05, 0.05) * pitch_y_px)
                center_x += dx
                center_y += dy

            # Place the ball on the height map
            y_start, y_end = center_y - patch_size//2, center_y + patch_size//2 + 1
            x_start, x_end = center_x - patch_size//2, center_x + patch_size//2 + 1

            if y_start < 0 or y_end > img_height_px or x_start < 0 or x_end > img_width_px:
                continue

            # Add ball profile to the substrate
            height_map[y_start:y_end, x_start:x_end] += current_ball_profile

    # All heights are in nanometers
    return height_map.astype(np.float32)


def simulate_interferometry(
    gt_height_map_nm,
    wavelengths_nm,
    num_buckets=3
):
    """
    Generates interferometry images from a ground truth height map.

    Args:
        gt_height_map_nm (np.ndarray): The ground truth height map in nanometers.
        wavelengths_nm (list or tuple): List of wavelengths in nanometers.
        num_buckets (int): Number of phase shift buckets.

    Returns:
        list: A list of generated intensity images (np.ndarray).
    """
    images = []
    A = 128  # Background intensity
    B = 100  # Modulation amplitude

    for wavelength in wavelengths_nm:
        # Calculate phase from height: phi = (4 * pi / lambda) * h
        phase = (4 * np.pi * gt_height_map_nm) / wavelength

        for b in range(num_buckets):
            phase_shift = b * (2 * np.pi / num_buckets)
            intensity = A + B * np.cos(phase + phase_shift)

            # Clip and convert to 8-bit grayscale image
            intensity_img = np.clip(intensity, 0, 255).astype(np.uint8)
            images.append(intensity_img)

    return images

def main(args):
    """Main function to generate the dataset."""

    output_dir = args.output_dir
    num_samples = args.num_samples

    # Create directories
    for i in range(num_samples):
        sample_name = f"sample_{i:03d}"
        raw_path = os.path.join(output_dir, sample_name, "raw")
        gt_path = os.path.join(output_dir, sample_name, "gt")
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

    print(f"Creating {num_samples} samples in '{output_dir}'...")

    # Physical parameters from the design document
    # Using a smaller image size for practical generation, but preserving aspect ratio
    # 9377x7000 is too large for typical local memory. Let's use a scaled down version.
    # The user can override this with command line arguments.
    img_width = args.img_width
    img_height = int(img_width * (7000 / 9377))

    # Calculate pixels_per_um based on the full image size and a hypothetical FOV.
    # Let's assume the 9377px width corresponds to a ~1mm field of view.
    # This gives a resolution of ~9.4 pixels/um.
    pixels_per_um = args.pixels_per_um

    for i in tqdm(range(num_samples), desc="Generating Samples"):
        sample_name = f"sample_{i:03d}"

        # 1. Generate Ground Truth Height Map
        gt_height_map = create_bga_height_map(
            img_shape=(img_height, img_width),
            ball_diameter_um=args.ball_diameter,
            ball_height_um=args.ball_height,
            ball_pitch_um=(args.pitch_y, args.pitch_x),
            pixels_per_um=pixels_per_um,
            randomize=args.randomize
        )

        # Save the GT map
        gt_filepath = os.path.join(output_dir, sample_name, "gt", "height.npy")
        np.save(gt_filepath, gt_height_map)

        # 2. Simulate Interferometry
        intensity_images = simulate_interferometry(
            gt_height_map_nm=gt_height_map,
            wavelengths_nm=args.wavelengths,
            num_buckets=args.num_buckets
        )

        # 3. Save Images
        raw_path = os.path.join(output_dir, sample_name, "raw")
        file_pattern = args.file_pattern
        num_wavelengths = len(args.wavelengths)

        for w_idx in range(num_wavelengths):
            for b_idx in range(args.num_buckets):
                img_idx = w_idx * args.num_buckets + b_idx
                img_filename = file_pattern.format(w_idx=w_idx, b_idx=b_idx)
                img_filepath = os.path.join(raw_path, img_filename)
                cv2.imwrite(img_filepath, intensity_images[img_idx])

    print("Synthetic BGA data generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic BGA Interferometry Data.")
    parser.add_argument('--output-dir', type=str, default="data/bga_synthetic",
                        help="Root directory to save the generated data.")
    parser.add_argument('--num-samples', type=int, default=10,
                        help="Number of data samples to generate.")
    parser.add_argument('--img-width', type=int, default=938,
                        help="Width of the generated images in pixels.")
    parser.add_argument('--pixels-per-um', type=float, default=2.0,
                        help="Resolution in pixels per micrometer.")

    # BGA physical properties
    parser.add_argument('--ball-diameter', type=float, default=100.0, help="Ball diameter in um.")
    parser.add_argument('--ball-height', type=float, default=50.0, help="Ball height in um.")
    parser.add_argument('--pitch-x', type=float, default=200.0, help="Pitch in X direction in um.")
    parser.add_argument('--pitch-y', type=float, default=200.0, help="Pitch in Y direction in um.")

    # Interferometry properties
    parser.add_argument('--wavelengths', type=float, nargs='+', default=[633.0, 532.0, 450.0, 780.0],
                        help="List of wavelengths in nm.")
    parser.add_argument('--num-buckets', type=int, default=3, help="Number of phase shift buckets.")
    parser.add_argument('--file-pattern', type=str, default="w{w_idx}_b{b_idx}.png",
                        help="Filename pattern for the output images.")

    # Other
    parser.add_argument('--randomize', action='store_true',
                        help="Enable domain randomization for ball properties.")

    args = parser.parse_args()
    main(args)
