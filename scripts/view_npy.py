import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def view_npy_file(file_path):
    """
    Loads and displays a 2D numpy array from a .npy file as an image.
    This is a utility script for data inspection and debugging.
    """
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다. '{file_path}'")
        return

    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"오류: .npy 파일을 로드하는 중 문제가 발생했습니다: {e}")
        return

    if data.ndim != 2:
        print(f"오류: .npy 파일에 2차원 배열이 없습니다. 현재 차원: {data.ndim}")
        return

    print(f"파일 시각화: {file_path}")
    print(f"배열 형태: {data.shape}, 데이터 타입: {data.dtype}")
    print(f"최소값: {np.min(data):.2f}, 최대값: {np.max(data):.2f}")

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='viridis')
    ax.set_title(os.path.basename(file_path))
    fig.colorbar(im, ax=ax, label='Height (nm)')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 2D numpy array from a .npy file.")
    parser.add_argument('file_path', type=str, help="Path to the .npy file to visualize.")
    args = parser.parse_args()
    view_npy_file(args.file_path)