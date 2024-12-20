from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from collections import defaultdict
from timeit import default_timer as timer

def create_writer(model_name: str,
                 experiment_name: str):
    """
    Needs testing. Please use it carefully.

    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance to track one specific experiment.

    Input Arguments:
    1) model_name = Name of the model.
    2) experiment_name = Name of the experiment.

    Output/Returns:
     - torch.utils.tensorboard.SummaryWriter() instance

    Example usuage:
    writer = create_writer(model_name="VisionTransformer",
                           experiment_name="5_epochs_with_SGD")
    """
    timestamp = datetime.now().strftime("&Y-%m-%d")
    log_dir = os.path.join("runs", timestamp, model_name, experiment_name)
    return SummaryWriter(log_dir=log_dir)

def accuracy_fn(y: torch.Tensor,
             y_hat: torch.Tensor):
    """
    Calculates the accuracy (one to one match).

    Input Arguments:
    1) y = Actual labels
    2) y_hat = Predicted labels

    Output/Returns:
     - Accuracy (float) between y and y_hat

    Example usuage:
    acc = accuracy_fn(y, y_hat)
    """
    assert y.shape == y_hat.shape, f"Shape mismatch. Please check. {y.shape} is not same as {y_hat.shape}"
    count = torch.eq(y_hat, y).sum().item()
    return count/len(y)

def train_step(model: nn.Module,
              train_dataLoader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy_fn,
              device: str):
    """
    Performs a singe train step for one epoch in the training loop. This function is a helper fucntion for fit() method.

    Input Arguments:
    1) model = Model being trained.
    2) train_dataLoader = The train dataloader containing the training image and label batches.
    3) loss_fn = Loss function used to train the model.
    4) optimizer = Optimizer being used to optimize the model parameters during gradient descent.
    5) accuracy_fn = A function which calculates the accuracy.
    6) device = A string denoting the device ("cpu" or "cuda") being used.

    Output/Returns:
     - train_loss (float) = Average training loss for the epoch.
     - train_acc (float) = Average training accuracy for the epoch.

    Example usuage:
    train_loss, train_acc = train_step(model=your_model,
                                       train_dataLoader=train_dataLoader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       accuracy_fn=accuracy_fn,
                                       device=device)
    """

    model.train()
    train_loss, train_acc = 0.0, 0.0 

    for X, y in tqdm(train_dataLoader):
        X, y = X.to(device), y.to(device)
        raw_logits = model(X)
        loss = loss_fn(raw_logits, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y, torch.argmax(raw_logits, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataLoader)
    train_acc /= len(train_dataLoader)

    return train_loss, train_acc

def val_step(model: nn.Module,
              val_dataLoader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              accuracy_fn,
              device: str):
    """
    Performs a singe val step for one epoch in the training loop. This function is a helper fucntion for fit() method.

    Input Arguments:
    1) model = Model being evaluated.
    2) val_dataLoader = The val dataloader containing the training image and label batches.
    3) loss_fn = Loss function used to train the model.
    4) accuracy_fn = A function which calculates the accuracy.
    5) device = A string denoting the device ("cpu" or "cuda") being used.

    Output/Returns:
     - val_loss (float) = Average validation loss for the epoch.
     - val_acc (float) = Average validation accuracy for the epoch.

    Example usuage:
    val_loss, val_acc = val_step(model=your_model,
                                       val_dataLoader=val_dataLoader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       accuracy_fn=accuracy_fn,
                                       device=device)
    """

    model.eval()

    val_loss, val_acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in tqdm(val_dataLoader):
            X, y = X.to(device), y.to(device)
            raw_logits = model(X)
            loss = loss_fn(raw_logits, y)
            val_loss += loss.item()
            val_acc = accuracy_fn(y, torch.argmax(raw_logits, dim=1))

    val_loss /= len(val_dataLoader)
    val_acc /= len(val_dataLoader)

    return val_loss, val_acc

def fit(model: nn.Module,
        train_dataLoader: torch.utils.data.DataLoader, 
        val_dataLoader: torch.utils.data.DataLoader,
        epochs: int,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        accuracy_fn=accuracy_fn,
        scheduler: torch.optim.Optimizer = None,
        writer: torch.utils.tensorboard.writer.SummaryWriter = None):
    """
    Performs training of the model with the given data, loss function and optimizer and other hyperparameters.

    Input Arguments:
    1) model = Model being trained
    2) train_dataLoader = DataLoader containing training batches.
    3) val_dataLoader = DataLoader containing validation batches.
    4) epochs = An integer denoting the number of epochs for training.
    5) loss_fn = Loss function to train the model.
    6) optimizer = Optimizer being used to optimize the model parameters during gradient descent.
    7) accuracy_fn = A function which calculates the accuracy.
    8) device = A string denoting the device ("cpu" or "cuda") being used.

    9) (Optional) scheduler = Learning Rate Scheduler applied after every epoch. Default = None.
    10) writer = Used to track experiments for Tensorboard. Default = None.

    Output/Returns:
     - A dictionary containing the training history namely the training and validation losses and accuracy for every epoch.
     Keys of the dictionary = "train_loss", "train_acc", "val_loss", "val_acc"
    """

    history = defaultdict(list)
    start_timer = timer()
    for epoch in tqdm(range(epochs)):
        print(f"Epoch = {epoch+1}")
        train_loss, train_acc = train_step(model=model,
                                         train_dataLoader=train_dataLoader,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         accuracy_fn=accuracy_fn,
                                         device=device)
        val_loss, val_acc = val_step(model=model,
                                    val_dataLoader=val_dataLoader,
                                    loss_fn=loss_fn,
                                    accuracy_fn=accuracy_fn,
                                    device=device)
        if scheduler is not None:
            print(f"Current Learning Rate = {scheduler.get_last_lr()[0]}")
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train loss = {train_loss: .2f}, Train accuracy = {train_acc*100.0: .2f}%")
        print(f"Val loss = {val_loss: .2f}, Val accuracy = {val_acc*100.0: .2f}%")
        print("_________________________________________________________________________________________________________________________________________")
        print("_________________________________________________________________________________________________________________________________________\n")
        
        """Tensorboard part here. Needs testing. Please use carefully."""
        if writer is not None:
            writer.add_scalars(main_tag="Loss", 
                              tag_scalar_dict={"train_loss" : train_loss,
                                              "val_loss" : val_loss},
                              global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                              tag_scalar_dict={"train_acc" : train_acc,
                                              "val_acc" : val_acc},
                              global_step=epoch)
            writer.close()
    end_timer = timer()
    print(f"Total time taken for training = {end_timer-start_timer: .2f} seconds.")
    return history