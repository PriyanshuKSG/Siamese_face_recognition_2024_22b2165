import torchvision
from torchvision import datasets
from matplotlib import pyplot as plt
import math
import torch
import os
from typing import List, Union, Sequence


def get_dataLoaders_and_class_names(train_dir: str,
                                    test_dir: str,
                                    train_transform: torchvision.transforms.Compose,
                                    test_transform: torchvision.transforms.Compose,
                                    batch_size: int,
                                    val_dir: str = None,
                                    num_workers: int = 0):
    """
    Creates train, test, and optional validation DataLoaders, and returns class names.

    This function works with the torchvision.datasets.ImageFolder format, expecting 
    directory structures where subdirectories represent class labels.
    
    Input Arguments:
    1) train_dir (str): Path to the training directory.
    2) test_dir (str): Path to the testing directory.
    3) train_transform (torchvision.transforms.Compose): Transforms for training images.
    4) test_transform (torchvision.transforms.Compose): Transforms for test/validation images.
    5) batch_size (int): Batch size for all DataLoaders.

    6) (Optional) val_dir (str): Path to the validation directory. Default = None.
    7) (Optional) num_workers (int): Number of subprocesses to load data in parallel. Default = 0.
    

    Output/Returns:
     - train_dataLoader = torch.utils.data.DataLoader
     - test_dataLoader = torch.utils.data.DataLoader
     - (optional) val_dataLoader = torch.utils.data.DataLoader
     - class_names = list

    Example usuage:
    train_loader, test_loader, class_names = get_dataLoaders_and_class_names(
    train_dir='path/to/train',
    test_dir='path/to/test',
    train_transform=train_transforms,
    test_transform=test_transforms,
    batch_size=32)
    """

    assert os.path.exists(train_dir), f"Training directory '{train_dir}' not found."
    assert os.path.exists(test_dir), f"Testing directory '{test_dir}' not found."
    if val_dir:
        assert os.path.exists(val_dir), f"Validation directory '{val_dir}' not found."
    
    train_ds = datasets.ImageFolder(root=train_dir,
                                   transform=train_transform)
    test_ds = datasets.ImageFolder(root=test_dir,
                                   transform=test_transform)
    print(f"Number of images in train dataset = {len(train_ds)}")
    print(f"Number of images in test dataset = {len(test_ds)}")

    if val_dir is not None:
        val_ds = datasets.ImageFolder(root=val_dir,
                                   transform=test_transform)
        print(f"Number of images in val dataset = {len(val_ds)}")
   
    class_names = train_ds.classes
    
    train_dataLoader = torch.utils.data.DataLoader(dataset=train_ds,
                                                  batch_size=batch_size,
                                                   num_workers=num_workers,
                                                  shuffle=True)
    test_dataLoader = torch.utils.data.DataLoader(dataset=test_ds,
                                                  batch_size=batch_size,
                                                   num_workers=num_workers,
                                                  shuffle=False)
    
    print("Successfully got all data loaders and class names!!")

    if val_dir is not None:
        val_dataLoader = torch.utils.data.DataLoader(dataset=val_ds,
                                                  batch_size=batch_size,
                                                   num_workers=num_workers,
                                                  shuffle=False)
        return train_dataLoader, test_dataLoader, val_dataLoader, class_names
    else:
        return train_dataLoader, test_dataLoader, class_names


def visualize_images_from_dataLoader(dataLoader: torch.utils.data.DataLoader,
                                     class_names: List[str],
                                     num_images: int,
                                    figsize: Union[int, Sequence[int]] = 10):
    """
    Given the train and test data loaders, this function visualizes images by plotting them using 
    matplotlib. 

    Input Arguments:
    1) dataLoader = A torch.utils.data.DataLoader conatining batches of images and labels.
    2) class_names = A list containing all the class names.
    3) num_images = An integer denoting the number of images to be plotted/visualized.

    4) (Optional) figsize = An integer or a sequence (list or tuple) used as a parameter in plt.subplots 

    Output/Returns:
     - Nothing. 

    Example usuage:
    visualize_images_from_dataLoader(dataLoader=train_dataLoader,
                                    class_names=class_names, 
                                    num_images=9)
    visualize_images_from_dataLoader(dataLoader=test_dataLoader,
                                    class_names=class_names, 
                                    num_images=13,
                                    figsize=(12,8))
    """
 
    assert(num_images > 1), "num_images must be greater than 1"
    assert(num_images <= 16), "num_images must be less than 16."

    if isinstance(figsize, int):
        figsize = (figsize, figsize)
    elif isinstance(figsize, (list, tuple)) and len(figsize) == 2:
        figsize = tuple(figsize)
    else:
        raise ValueError("figsize must be an integer or a sequence (width, height).")

    n = int(math.ceil(math.sqrt(num_images)))
    fig, axs = plt.subplots(n, n, figsize=figsize)
    it = iter(dataLoader)
    image_batch, label_batch = next(it)
    axs = axs.flatten()
    
    print("Visualizing images")
    for i in range(num_images):
        index = torch.randint(0, len(image_batch), size=[1]).item()
        image, label = image_batch[index], label_batch[index]
        assert image.shape[0] == 3, "Image must have color channels first (C, H, W). Use torch.permute if needed."
        axs[i].imshow(image.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
        axs[i].set_title(class_names[label.item()])
        axs[i].axis('off')

    for j in range(num_images, len(axs)):
        axs[j].axis('off')
        
    plt.show()

def plot_loss(history: dict):
    """
    Plots the train and val losses and accuracies.

    Input Arguments:
    1) history = A dictionary conatining training details of the model.

    Output/Returns:
     - Nothing

    Example usuase:
    plot_loss(history)
    """

    fig, axs = plt.subplots(1,2,figsize=(8,4))
    axs[0].plot([i for i in range(len(history['train_loss']))], history['train_loss'], color='blue', label="train loss")
    axs[0].plot([i for i in range(len(history['val_loss']))], history['val_loss'], color='orange', label="val loss")
    axs[1].plot([i for i in range(len(history['train_acc']))], history['train_acc'], color='blue', label="train accuracy")
    axs[1].plot([i for i in range(len(history['val_acc']))], history['val_acc'], color='orange', label="val accuracy")
    axs[0].set_title("Loss curves")
    axs[1].set_title("Accuracy curves")
    axs[0].legend()
    axs[1].legend()
    plt.show()