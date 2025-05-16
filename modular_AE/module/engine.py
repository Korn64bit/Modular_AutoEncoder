import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
from utils import save_model

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: callable,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               is_vae: bool = False,
               recon_loss_type_vae: str = 'BCE') -> Tuple[float, float]:

    model.train()

    train_loss = 0

    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        images_flat = images.view(images.size(0), -1)
        optimizer.zero_grad()

        if is_vae:
            recon_images_flat, mu, log_var = model(images_flat)
            loss = loss_fn(recon_images_flat, images_flat, mu, log_var, recon_loss_type=recon_loss_type_vae)
        else:
            recon_images_flat = model(images_flat)
            loss = loss_fn(recon_images_flat, images_flat)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

    total_train_loss = train_loss / len(dataloader.dataset)

    return total_train_loss

def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: callable,
             device: torch.device,
             is_vae: bool = False,
             recon_loss_type_vae: str = 'BCE') -> Tuple[float, float]:
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for images, _ in dataloader:
            images = images.to(device)
            images_flat = images.view(images.size(0), -1)

            if is_vae:
                recon_images_flat, mu, log_var = model(images_flat)
                loss = loss_fn(recon_images_flat, images_flat, mu, log_var, recon_loss_type=recon_loss_type_vae)
            else:
                recon_images_flat = model(images_flat)
                loss = loss_fn(recon_images_flat, images_flat)

            val_loss += loss.item() * images.size(0)
    total_val_loss = val_loss / len(dataloader.dataset)
    return total_val_loss

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: callable,
              device: torch.device,
              is_vae: bool = False,
              recon_loss_type_vae: str = 'BCE') -> Tuple[float, float]:
    # put model in eval mode
    model.eval()

    # setup test loss and test accuracy value
    test_loss = 0

    # turn on inference context manager
    with torch.inference_mode():
        # loop through DataLoader batches
        for images, _ in dataloader:
            images = images.to(device)
            images_flat = images.view(images.size(0), -1)

            if is_vae:
                recon_images_flat, mu, log_var = model(images_flat)
                loss = loss_fn(recon_images_flat, images_flat, mu, log_var, recon_loss_type=recon_loss_type_vae)
            else:
                recon_images_flat = model(images_flat)
                loss = loss_fn(recon_images_flat, images_flat)

            test_loss += loss.item() * images.size(0)

    # adjust metrics to get average loss and accuracy per batch
    total_test_loss = test_loss / len(dataloader.dataset)

    # print(f"Test Loss: {total_test_loss:.6f}")
    return total_test_loss

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: callable,
          epochs: int,
          device: torch.device,
          patience: int = 5,
          min_delta: float = 0.001,
          model_save_dir: str = 'trained_models',
          model_save_name: str = 'best_model.pth',
          is_vae: bool = False,
          recon_loss_type_vae: str = 'BCE') -> Dict[str, List]:
    # create empty results dictionary
    results = {'train_loss': [],
               'val_loss': [],
               'test_loss': []}
    best_val_loss = np.inf
    epochs_no_improve = 0

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    best_model_save_path = Path(model_save_dir) / model_save_name
    
    # loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device,
                                is_vae=is_vae,
                                recon_loss_type_vae=recon_loss_type_vae)
        val_loss = val_step(model=model,
                            dataloader=val_dataloader,
                            loss_fn=loss_fn,
                            device=device,
                            is_vae=is_vae,
                            recon_loss_type_vae=recon_loss_type_vae)

        test_loss = test_step(model=model,
                            dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            device=device,
                            is_vae=is_vae,
                            recon_loss_type_vae=recon_loss_type_vae)
        
        # print out what's happening
        print(f'Epoch: {epoch+1} | '
              f'train_loss: {train_loss:.4f} | '
              f'val_loss: {val_loss:.4f} | '
              f'test_loss: {test_loss:.4f} | ')
        
        # update results dictionary
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['test_loss'].append(test_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            if 'save_model' in globals() and callable(globals()['save_model']):
                print(f'[INFO] Validation loss improves. Saving model to {best_model_save_path}')
                torch.save(obj=model.state_dict(), f=best_model_save_path)
            else:
                print(f'[INFO] Validation loss improved, but save_model function is not available to save the best model.')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'[INFO] Early stopping triggered after {epoch+1} due to no improvement in validation loss')
            break

    # return teh filled results at the end of the epochs val_loss
    return results

def vae_loss_fn(recon_x, x, mu, log_var, recon_loss_type='BCE'):
    if recon_loss_type == 'BCE':
        recon_loss_fn = nn.BCELoss(reduction='sum')
        recon_loss = recon_loss_fn(recon_x, x.view(-1, 28*28))
    elif recon_loss_type == 'MSE':
        recon_loss_fn = nn.MSELoss(reduction='sum')
        recon_loss = recon_loss_fn(recon_x, x.view(-1, 28*28))
    else:
        raise ValueError('Unsupported reconstruction_loss_type. Choose "BCE" or "MSE".')
    
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kld