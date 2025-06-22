import numpy as np
import cv2
import os
from tqdm import tqdm

def generate_smooth_surface(shape=(256, 256), n_waves=5, max_height=500.0):
    """2D 사인파를 합쳐 부드러운 형태의 임의의 표면을 생성합니다."""
    height_map = np.zeros(shape, dtype=np.float32)
    for _ in range(n_waves):
        # 임의의 주파수, 진폭, 위상
        freq_x = np.random.rand() * 0.1 + 0.01
        freq_y = np.random.rand() * 0.1 + 0.01
        amp = np.random.rand() * 0.5 + 0.5
        phase_x = np.random.rand() * 2 * np.pi
        phase_y = np.random.rand() * 2 * np.pi

        x = np.linspace(0, 2 * np.pi, shape[1])
        y = np.linspace(0, 2 * np.pi, shape[0])
        xx, yy = np.meshgrid(x, y)

        wave = amp * np.sin(freq_x * xx + phase_x) * np.cos(freq_y * yy + phase_y)
        height_map += wave

    # [0, max_height] 범위로 정규화
    height_map -= height_map.min()
    height_map /= height_map.max()
    height_map *= max_height
    return height_map.astype(np.float32)

def generate_interferometry_data(
    output_dir="data/raw",
    num_samples=5,
    img_size=(256, 256),
    wavelengths=(633.0, 532.0, 450.0), # nm 단위
    max_height=800.0 # nm 단위
):
    """3파장 4-bucket 간섭계를 위한 가상 데이터셋을 생성합니다."""
    benchmark_path = os.path.join(output_dir, "benchmark")
    gt_path = os.path.join(output_dir, "benchmark_gt")

    os.makedirs(benchmark_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    print(f"'{output_dir}'에 {num_samples}개의 샘플을 생성합니다...")

    for i in tqdm(range(1, num_samples + 1), desc="샘플 생성 중"):
        sample_name = f"sample_{i:03d}"
        sample_dir = os.path.join(benchmark_path, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        # 1. 정답(Ground Truth) 높이 맵 생성
        gt_height_map = generate_smooth_surface(shape=img_size, max_height=max_height)
        
        # GT 맵을 float32 정밀도를 보존하는 numpy 배열(.npy)로 저장
        np.save(os.path.join(gt_path, f"{sample_name}_gt.npy"), gt_height_map)

        # 2. 높이 맵을 기반으로 12개의 intensity 이미지 생성
        A = 128  # 배경 밝기
        B = 100  # 변조 진폭

        for w_idx, wavelength in enumerate(wavelengths):
            # 높이로부터 위상 계산
            phase = (4 * np.pi * gt_height_map) / wavelength
            
            for b in range(1, 5): # 4-bucket
                phase_shift = (b - 1) * np.pi / 2.0
                intensity = A + B * np.cos(phase + phase_shift)
                
                # 8-bit 흑백 이미지로 변환하여 저장
                intensity = np.clip(intensity, 0, 255).astype(np.uint8)
                img_filename = f"L{w_idx+1}_I{b}.png"
                cv2.imwrite(os.path.join(sample_dir, img_filename), intensity)

    print("가상 데이터 생성이 완료되었습니다.")
    print(f"데이터 저장 위치: {benchmark_path}")
    print(f"정답(GT) 저장 위치: {gt_path}")

if __name__ == "__main__":
    generate_interferometry_data()