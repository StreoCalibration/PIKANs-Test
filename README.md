# 프로젝트 개요: Physics-Informed KAN for Interferometry (PIKANs)

## 1. 프로젝트 목표

본 프로젝트는 실제 측정 장비에서 Ground Truth 확보가 거의 불가능한 나노미터(nm) 스케일의 3D 높이 복원 문제를 해결합니다. 컴퓨터 시뮬레이션으로 생성한 가상 데이터로 모델을 사전학습하고, 물리 법칙(Physics-Informed Loss)에만 의존하여 정답 없이 실제 데이터에 적응시키는 'Sim-to-Real' 하이브리드 전략을 통해 기존의 한계를 돌파합니다.

## 2. 구현 기술

### 2.1. Sim-to-Real 하이브리드 전략

본 프로젝트의 핵심은 'Sim-to-Real' 접근 방식입니다. 이는 두 단계로 구성됩니다.

1.  **사전학습 (Pre-training)**: 제어된 환경의 컴퓨터 시뮬레이션으로 이상적인 간섭 무늬 이미지와 해당 높이 맵(Ground Truth)을 대량 생성합니다. 이 데이터를 사용하여 모델이 간섭계의 기본적인 물리적 원리와 BGA(Ball Grid Array)와 같은 특정 구조의 형상을 학습하도록 합니다.
2.  **미세조정 (Fine-tuning)**: 사전학습된 모델을 실제 측정 데이터에 적용합니다. 이 단계에서는 Ground Truth 데이터 없이, 오직 물리 법칙에 기반한 손실 함수(Physics-Informed Loss)만을 사용하여 모델을 최적화합니다. 이를 통해 시뮬레이션 데이터와 실제 데이터 간의 차이(예: 센서 노이즈, 광학계 왜곡)를 모델이 스스로 보정하고 적응하도록 합니다.

### 2.2. PIKANs (Physics-Informed Kolmogorov-Arnold Networks)

본 프로젝트를 위해 특별히 설계된 네트워크 아키텍처입니다.

-   **KAN (Kolmogorov-Arnold Network)**: 기존의 다층 퍼셉트론(MLP)이 고정된 활성화 함수를 사용하는 것과 달리, KAN은 학습 가능한 스플라인(Spline) 함수를 활성화 함수로 사용합니다. 이는 네트워크가 더 복잡하고 비선형적인 관계를 적은 파라미터로 학습할 수 있게 하여 높은 정확도와 해석 가능성을 제공합니다. 본 프로젝트에서는 `src/model/pikans.py` 와 `src/model/layers.py` 에서 KAN 아키텍처를 구현했습니다.

-   **Physics-Informed Loss**: 모델의 예측이 물리 법칙을 따르도록 강제하는 손실 함수입니다. 간섭계의 물리 방정식 `I = A + B * cos(phi + delta)`을 손실 함수에 직접 통합했습니다. 모델이 예측한 높이(`h`)로부터 위상(`phi`)을 계산하고, 이를 통해 재구성한 간섭 무늬와 실제 측정된 간섭 무늬의 차이를 최소화하도록 학습합니다. 이 방식 덕분에 Ground Truth 없이도 실제 데이터에 대한 학습이 가능합니다. 이 로직은 `src/loss/physics_informed_loss.py` 에 구현되어 있습니다.

## 3. 참고 문헌

-   **Kolmogorov-Arnold Networks**:
    -   Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov-Arnold Networks. *arXiv preprint arXiv:2404.19756*.
-   **Physics-Informed Neural Networks (PINNs)**:
    -   Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics, 378*, 686-707.
