# Hyperparameters and paths
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 100
    img_channels = 3  # 1 for MNIST, 3 for CelebA
    img_size = 64
    batch_size = 128
    lr = 0.0002
    epochs = 50
    beta1 = 0.5  # Adam optimizer parameter
    dataset_path = "data/train"  # Path to training images
    save_dir = "models/"  # Directory to save models
    output_dir = "outputs/"  # Directory to save generated images