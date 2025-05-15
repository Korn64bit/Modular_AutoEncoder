import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    model.train()

    train_loss = 0

    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)

        images_flat = images.view(images.size(0), -1)

        optimizer.zero_grad()

        recon_images_flat = model(images_flat)

        loss = loss_fn(recon_images_flat, images_flat)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

    total_train_loss = train_loss / len(dataloader)
    # print(f'    Batch {i+1}/{len(dataloader)}, Current Batch Loss: {loss.item():.6f}')
    return total_train_loss

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
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
            recon_flat = model(images_flat)
            loss = loss_fn(recon_flat, images_flat)
            test_loss += loss.item() * images.size(0)

    # adjust metrics to get average loss and accuracy per batch
    total_test_loss = test_loss / len(dataloader)

    # print(f"Test Loss: {total_test_loss:.6f}")
    return total_test_loss

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    # create empty results dictionary
    results = {'train_loss': [],
               'test_loss': []}
    
    # loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # print out what's happening
        print(f'Epoch: {epoch+1} | '
              f'train_loss: {train_loss:.4f} | '
              f'test_loss: {test_loss:.4f} | ')
        
        # update results dictionary
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

    # return teh filled results at the end of the epochs val_loss
    return results