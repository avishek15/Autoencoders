# Autoencoders — Deep Learning Implementations

**PyTorch implementations of various autoencoder architectures.**

## What It Does

Collection of autoencoder models for dimensionality reduction, anomaly detection, and generative tasks.

**Why it matters:** Autoencoders are foundational to modern deep learning. Understanding them unlocks many advanced techniques.

## Implementations

| Model | Use Case |
|-------|----------|
| **Vanilla Autoencoder** | Basic dimensionality reduction |
| **Denoising Autoencoder** | Noise removal, robust features |
| **Variational Autoencoder (VAE)** | Generative modeling |
| **Sparse Autoencoder** | Feature learning |

## Quick Start

\`\`\`bash
# Clone
git clone https://github.com/avishek15/Autoencoders.git
cd Autoencoders

# Install
pip install -r requirements.txt

# Train
python train.py --model vae --dataset mnist --epochs 50
\`\`\`

## Tech Stack

- **Python 3.8+**
- **PyTorch** — Deep learning
- **MNIST/CIFAR** — Datasets

## Results

| Model | Reconstruction Error | Latent Dim |
|-------|---------------------|------------|
| Vanilla AE | 0.012 | 32 |
| VAE | 0.018 | 32 |
| Denoising AE | 0.015 | 32 |

## License

MIT License

## Author

Built by [Avishek Majumder](https://invaritech.ai)
