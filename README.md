# Inpainting with Variational Autoencoders

This project was developed as a final assignment for the **Probabilistic Graphical Models** course. Its goal is to explore and compare the performance of **variational autoencoder-based models** for the **image inpainting task** — reconstructing missing or masked parts of an image. We focus specifically on reconstructing images of cats with randomly applied masks using three advanced VAE models:

* **Vanilla VAE** (as a baseline)
* **VQ-VAE**
* **AVAE**
* **VAE + TreeVI**

The primary dataset used for training and evaluation comes from Kaggle:

📦 [Cat Image Dataset on Kaggle](https://www.kaggle.com/datasets/mahmudulhaqueshawon/cat-image)

---

## 🧠 Inpainting Task Overview

Image inpainting is the task of reconstructing missing regions in an image. In this project, we simulate missing regions by applying random masks to parts of cat images and ask the models to predict and reconstruct the original image content.

---

## 🧪 How to Run

Make sure your working directory is at the root level (i.e., the parent of `src/`), then use the following commands:

### 0. Install the pre-requisites

Recommended to create venv first:

```bash
pip install -r requirements.txt
```

### 1. Run the ETL pipeline

This will run the entire pipeline of downloading dataset, prerocessing and running training of every model:

```bash
make reproduce
```

### Example: Train the baseline VAE model

This command trains a standard VAE to serve as a baseline:

```bash
make run_training
```

---

## 📂 Project Structure

```
project-root/
│
├── src/
│   ├── main.py                 # Entry point for training & experiments
│   ├── models/                 # All model definitions (VAE, VQ-VAE, etc.)
│   │    ├── baseline/          # Data loading and preprocessing logic
│   │    ├── gan/               # Implementation of GAN-based models
│   │    ├── ssl/               # Implementation of SSL-based models - Barlow, SimCLR, BYOL
│   │    ├── treevi/            # Implementation of TreeVI-based models
│   │    ├── vae/               # Implementation of VAE-based models
│   │    ├── vqvae/             # Implementation of VQ-VAE models
│   ├── scripts/
│   │   ├── etl_process/        # Data loading and preprocessing logic
│   │   ├── finetune/           # Fine-tuning scripts for SSL pre-trained models
│   │   ├── reconstruction/     # Training of reconstruction models
│   │   ├── ssl/                # Code for training SSL models
│   └── training/               # Training logic for all models + Masked Dataset implementation
├── data/                       # Processed data, model checkpoints, metrics
├── experiments/                # Playground for experiments, logs, and results
├── Makefile                    # Commands for running training/ETL
└── README.md
```

---

## 🤖 Model Descriptions

### 1. **Vanilla VAE** (Baseline)

The **Variational Autoencoder** (VAE) is a probabilistic generative model that learns to encode images into a latent space and decode them back. It is trained by maximizing a variational lower bound on the data likelihood using reconstruction loss (e.g., MSE) and a KL-divergence regularizer.

Used as a **baseline** due to its simplicity and strong theoretical foundations.

* Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

---

### 2. **VQ-VAE** (Vector Quantized VAE)

**VQ-VAE** replaces the continuous latent space of standard VAEs with a **discrete latent space**, using vector quantization. This helps overcome the common issue of blurry reconstructions seen in standard VAEs and improves the ability to model complex images by learning a dictionary of discrete latent embeddings.

* Advantages: Sharp outputs, discrete bottleneck improves generation
* Used because it can **better preserve image structures** in reconstruction

📄 Paper: [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)

---

### 3. **AVAE** (Auxiliary VAE)

**AVAE** introduces auxiliary latent variables to improve the flexibility of the approximate posterior distribution, making inference more expressive and reducing the gap between the true and variational posterior.

* Advantages: Richer latent representation, better reconstruction accuracy
* Used because it **boosts inference quality** by extending posterior capacity

📄 Paper: [Auxiliary Variational Autoencoders](https://arxiv.org/abs/2012.11551)

---

### 4. **TreeVI + VAE**

**TreeVI** is a variational inference method that builds a **tree-structured approximation** to the posterior, improving marginal likelihood estimates and enabling **hierarchical reasoning** in the latent space.

* Advantages: More accurate posterior, better uncertainty modeling
* Used to test **cutting-edge hierarchical inference** with VAE models

📄 Paper: [Tree Variational Inference](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1a63a6a092a95bd45f0237766ac878ba-Abstract-Conference.html)

---

---

## 🧑‍🎓 Authors

This project was developed as part of coursework for **Probabilistic Graphical Models** at **Wrocław University of Science and Technology**.

---
