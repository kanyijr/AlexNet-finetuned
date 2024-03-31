import torchvision, torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm.auto import tqdm
from torch import nn

def create_dataloader(train_dir:str,
                      test_dir:str,
                      transform:transforms.Compose,
                      BATCH_SIZE:int,
                      NUM_WORKERS:int):
    


    train_data = ImageFolder(root=train_dir, transform=transform, target_transform=None)
    test_data = ImageFolder(root=test_dir, target_transform=None, transform=transform)


    train_dataloader = DataLoader(dataset=train_data, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    class_names = train_data.classes

    return train_dataloader, test_dataloader, class_names






