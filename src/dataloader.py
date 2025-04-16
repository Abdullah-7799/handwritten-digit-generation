from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(Config.img_size),
        transforms.CenterCrop(Config.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # For MNIST: (0.5,), for CelebA: (0.5, 0.5, 0.5)
    ])
    dataset = datasets.ImageFolder(root=Config.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    return dataloader