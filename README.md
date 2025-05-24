# MNI-GAIR: Multi-scale Normal Image and Grid Attention-based Image Recognition

![Model Architecture](MNI-GAIR-overview.png)  
*Figure: Overview of the MNI-GAIR framework*

**MNI-GAIR** is a state-of-the-art multimodal collaborative framework for high-precision facial recognition in complex scenarios (e.g., occlusion, extreme poses, and lighting variations). It integrates **multi-scale normal maps**, **dynamic grid attention mechanisms**, and **Lie group-based point cloud generalization** to achieve robust 3D facial analysis.  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Paper Link](https://img.shields.io/badge/PDF-Paper-red)]()  
[![GitHub Stars](https://img.shields.io/github/stars/yassinza/D2TS?style=social)](https://github.com/LLxuLL/MNI-GAIR-Multi-scale-Normal-Image-and-Grid-Attention-based-Image-Recognition)

---

## ✨ Key Features

- **Multi-scale Normal Map Generation**  
  Combines GLCM-LBP texture descriptors and cascaded atrous pyramids (CAP) to enhance geometric representation. Achieves **23.6% higher noise robustness** on CelebA-HQ.

- **Dynamic Grid Partitioning Attention Network (DGPA-Net)**  
  Implements gradient-driven grid optimization and dual-path attention (channel + spatial) to improve recognition accuracy for extreme profiles (>75°) by **14.7%** on LFW.

- **Point Cloud Generalization**  
  Utilizes FPFH-SVD coarse registration and Geman-McClure fine optimization, reducing cross-pose EER to **1.23%** on FaceScape.

- **Real-Time Efficiency**  
  End-to-end inference time of **46.7 ms** on A100 GPU with multi-threaded CUDA parallelism.

---

## 🚀 Performance Highlights

| Metric                | FaceScape (NRE↓) | LFW (EER↓) | 300W-LP (F1↑) | Inference Time (ms) |
|-----------------------|------------------|------------|---------------|---------------------|
| **MNI-GAIR (Ours)**   | **0.082**        | **1.23%**  | **0.96**      | **46.7**           |
| 3DDFA_V2              | 0.131            | 2.95%      | 0.87          | 45.3               |
| 3D-PointMap           | 0.095            | 1.87%      | -             | 38.9               |

---

## 📥 Installation

### Prerequisites
- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA ≥ 11.3
- Open3D ≥ 0.15

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yassinza/D2TS.git
   cd D2TS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pretrained weights:
   ```bash
   wget https://github.com/yassinza/D2TS/releases/download/v1.0/point_cloud_facescape_model.pth -P ./saved_models/
   ```

---

## 🛠️ Quick Start

### Inference on a Single Image
```bash
from point_cloud_predict import predict_and_save_point_cloud

results = predict_and_save_point_cloud()  # Results saved to `saved_files/point_cloud_predictions.json`
print("Top-5 Predictions:", results)
```

### Training
```bash
python point_cloud_train.py --dataset_path ./datasets/Points/facescape_dataset
```

---

## 📖 Citation

If you use MNI-GAIR in your research, please cite:
```bibtex
@article{xu2025mni,
  title={MNI-GAIR: Multi-scale Normal Image and Grid Attention-based Image Recognition},
  author={Xu, Maoyang and Zheng, Zhuqing and Zhang, Changjiang etc.},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request for any improvements.  
For questions, contact [Maoyang Xu](mailto:18810775071@163.com) or [Changjiang Zhang](mailto:zhangchangjiang@ccbupt.cn).

--- 

## 🌐 Applications
- **Security Surveillance**: Real-time face recognition under occlusion/low-light conditions.  
- **Metaverse Avatars**: 3D face reconstruction from monocular RGB images.  
- **Medical Analysis**: Micro-expression detection for diagnostic support.

---
