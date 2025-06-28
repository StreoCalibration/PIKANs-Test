import numpy as np
import os
import argparse
from PIL import Image

def generate_ground_truth_height_map(width, height):
    """
    가우시안 피크, 사인파, 기울어진 평면을 조합하여 복잡한 형태의 Ground Truth 높이 맵을 생성합니다.
    단위는 나노미터(nm)입니다.
    """
    x = np.linspace(-5, 5, width)
    y = np.linspace(-5, 5, height)
    xx, yy = np.meshgrid(x, y)

    # 다양한 형태의 표면 조합
    peak = 150 * np.exp(-((xx - 1.5)**2 + (yy - 1.5)**2) / 4)
    wave = 50 * np.sin(2 * xx) * np.cos(2 * yy)
    plane = 10 * xx + 5 * yy

    height_map = peak + wave + plane + 50  # 기본 높이 추가
    # 약간의 노이즈 추가
    noise = np.random.normal(0, 2, (height, width))
    height_map += noise

    return height_map.astype(np.float32)

def simulate_interferometry(height_map, wavelengths, phase_shifts, output_dir):
    """
    4-파장, 3-bucket 간섭계 측정 과정을 시뮬레이션하고 결과 이미지를 저장합니다.
    물리식: I = A + B * cos( (4*pi/lambda) * h + delta )
    """
    height, width = height_map.shape

    # 배경(A) 및 변조(B) 강도 (약간의 불균일성 포함)
    A = 128 + np.random.normal(0, 5, (height, width))
    B = 100 + np.random.normal(0, 5, (height, width))

    image_filenames = []

    for i, wl in enumerate(wavelengths):
        # 높이로부터 위상(phase) 계산
        phi = (4 * np.pi / wl) * height_map

        for j, delta in enumerate(phase_shifts):
            # 현재 bucket의 간섭 강도 계산
            intensity = A + B * np.cos(phi + delta)

            # 측정 노이즈(shot noise) 추가
            intensity += np.random.normal(0, 2, intensity.shape)

            # 8-bit 이미지로 변환 (0-255)
            intensity_clipped = np.clip(intensity, 0, 255)
            intensity_uint8 = intensity_clipped.astype(np.uint8)

            # 이미지 파일로 저장
            img = Image.fromarray(intensity_uint8, 'L')
            # 파일명 규칙: w{파장인덱스}_b{버킷인덱스}.png (e.g., w0_b0.png)
            filename = f"w{i}_b{j}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            image_filenames.append(filename)

    print(f"생성 완료: {len(image_filenames)}개의 간섭 이미지가 {output_dir}에 저장되었습니다.")
    return image_filenames

def main():
    parser = argparse.ArgumentParser(description="4-파장, 3-bucket 간섭계 시뮬레이션 데이터 생성기")
    parser.add_argument('--output_dir', type=str, default='data/4w3b_benchmark', help="생성된 데이터를 저장할 기본 디렉터리")
    parser.add_argument('--num_samples', type=int, default=5, help="생성할 데이터 샘플의 수")
    parser.add_argument('--width', type=int, default=256, help="이미지 너비")
    parser.add_argument('--height', type=int, default=256, help="이미지 높이")
    args = parser.parse_args()

    # 물리 파라미터 정의
    # 4개의 파장 (단위: nm)
    wavelengths = [450.0, 532.0, 632.8, 780.0]
    # 3-bucket 위상 변이량 (단위: radian)
    phase_shifts = [0, 2 * np.pi / 3, 4 * np.pi / 3]

    for i in range(args.num_samples):
        sample_name = f"sample_{i:03d}"
        sample_dir = os.path.join(args.output_dir, sample_name)
        gt_dir = os.path.join(sample_dir, "gt")
        raw_data_dir = os.path.join(sample_dir, "raw")

        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(raw_data_dir, exist_ok=True)

        print(f"\n--- 샘플 생성 중: {sample_name} ---")

        # 1. Ground Truth 높이 맵 생성 및 저장
        gt_height_map = generate_ground_truth_height_map(args.width, args.height)
        gt_filepath = os.path.join(gt_dir, "height.npy")
        np.save(gt_filepath, gt_height_map)
        print(f"Ground Truth 높이 맵 저장 완료: {gt_filepath}")

        # 2. 간섭 무늬 이미지 시뮬레이션 및 저장
        simulate_interferometry(gt_height_map, wavelengths, phase_shifts, raw_data_dir)

    print(f"\n총 {args.num_samples}개의 샘플을 '{args.output_dir}'에 성공적으로 생성했습니다.")

if __name__ == "__main__":
    main()