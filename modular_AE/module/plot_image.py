import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def plot_reconstructed_images(model, dataloader, device, num_images_to_plot=10):
    """
    Plots a specified number of original and reconstructed images from the dataloader,
    with row titles on the far left using fig.text().

    Args:
        model (torch.nn.Module): The trained autoencoder model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): The device the model and data are on ('cuda' or 'cpu').
        num_images_to_plot (int): Number of images to plot.
    """
    model.eval()
    
    images, _ = next(iter(dataloader))
    images = images.to(device)

    if images.size(0) < num_images_to_plot:
        print(f"Warning: Requested {num_images_to_plot} images, but batch size is {images.size(0)}. Plotting {images.size(0)} images instead.")
        num_images_to_plot = images.size(0)

    original_images = images[:num_images_to_plot]
    original_images_flat = original_images.view(original_images.size(0), -1)
    
    with torch.no_grad():
        reconstructed_images_flat = model(original_images_flat)
    
    reconstructed_images = reconstructed_images_flat.view(original_images.size(0), 1, 28, 28)

    original_images = original_images.cpu()
    reconstructed_images = reconstructed_images.cpu()

    # Create the figure and subplots. We can use subplots() directly for more control.
    # This also makes it easier to get the positions of the axes for fig.text()
    fig, axes = plt.subplots(2, num_images_to_plot, figsize=(num_images_to_plot * 1.8, 4.0))
    # Slightly adjusted figsize and multiplier for potentially better default spacing

    for i in range(num_images_to_plot):
        # Plot original images
        axes[0, i].imshow(original_images[i].squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')

        # Plot reconstructed images
        axes[1, i].imshow(reconstructed_images[i].squeeze().numpy(), cmap='gray')
        axes[1, i].axis('off')
    
    # Add titles using fig.text()
    # These coordinates are in figure fraction (0,0 is bottom left, 1,1 is top right)
    # We need to estimate a good y-position.
    # The first row of subplots is roughly centered around y=0.75 (can vary)
    # The second row of subplots is roughly centered around y=0.25 (can vary)
    
    # To make it more robust, we can get the position of the first axes in each row
    # and use that to guide the y-coordinate of the fig.text
    
    # Get the y-coordinate of the center of the first row of axes
    # Position is [left, bottom, width, height]
    y_pos_row1 = axes[0, 0].get_position().y0 + axes[0, 0].get_position().height / 2
    y_pos_row2 = axes[1, 0].get_position().y0 + axes[1, 0].get_position().height / 2

    # Add text to the figure. Adjust x coordinate (e.g., 0.03) to control distance from left edge.
    fig.text(0.03, y_pos_row1, "Original", fontsize=12, rotation=0, 
             verticalalignment='center', horizontalalignment='left')
    fig.text(0.03, y_pos_row2, "Reconstructed", fontsize=12, rotation=0, 
             verticalalignment='center', horizontalalignment='left')

    # Adjust subplot parameters to prevent overlap with fig.text and ensure good layout
    # You might need to fine-tune 'left' to ensure text is visible and 'wspace'/'hspace' for image spacing.
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.02, wspace=0.1, hspace=0.1)
    
    plt.show()

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def plot_reconstructed_images_vae(model, dataloader, device, num_images_to_plot=10, is_vae=False): # Added is_vae flag
    """
    Plots a specified number of original and reconstructed images from the dataloader,
    with row titles on the far left using fig.text().

    Args:
        model (torch.nn.Module): The trained autoencoder model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): The device the model and data are on ('cuda' or 'cpu').
        num_images_to_plot (int): Number of images to plot.
        is_vae (bool): Set to True if the model is a Variational Autoencoder,
                       which returns (reconstruction, mu, log_var).
    """
    model.eval() # Set model to evaluation mode
    
    # Get a batch of images
    images, _ = next(iter(dataloader))
    images = images.to(device)

    # Ensure we don't try to plot more images than available in the batch
    if images.size(0) < num_images_to_plot:
        print(f"Warning: Requested {num_images_to_plot} images, but batch size is {images.size(0)}. Plotting {images.size(0)} images instead.")
        num_images_to_plot = images.size(0)

    original_images = images[:num_images_to_plot]
    original_images_flat = original_images.view(original_images.size(0), -1) # Flatten original images
    
    with torch.no_grad(): # No need to track gradients for plotting
        model_output = model(original_images_flat) # Get model output
        
        # If the model is a VAE, it returns a tuple (recon, mu, log_var)
        # We only need the reconstruction (the first element)
        if is_vae:
            reconstructed_images_flat = model_output[0]
        else:
            reconstructed_images_flat = model_output
    
    # Reshape reconstructed images back to image dimensions (e.g., 1x28x28 for FashionMNIST)
    # Assuming single channel (grayscale) images
    reconstructed_images = reconstructed_images_flat.view(original_images.size(0), 1, 28, 28)

    # Move images to CPU for plotting with matplotlib
    original_images = original_images.cpu()
    reconstructed_images = reconstructed_images.cpu()

    # Create the figure and subplots
    fig, axes = plt.subplots(2, num_images_to_plot, figsize=(num_images_to_plot * 1.8, 4.5)) # Adjusted figsize slightly
    # Ensure axes is always 2D, even if num_images_to_plot is 1
    if num_images_to_plot == 1:
        axes = axes.reshape(2, 1)


    for i in range(num_images_to_plot):
        # Plot original images
        axes[0, i].imshow(original_images[i].squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')

        # Plot reconstructed images
        axes[1, i].imshow(reconstructed_images[i].squeeze().numpy(), cmap='gray')
        axes[1, i].axis('off')
    
    # Add titles using fig.text()
    # To make it more robust, get the position of the first axes in each row
    y_pos_row1 = axes[0, 0].get_position().y0 + axes[0, 0].get_position().height / 2
    y_pos_row2 = axes[1, 0].get_position().y0 + axes[1, 0].get_position().height / 2

    fig.text(0.03, y_pos_row1, "Original", fontsize=12, rotation=0, 
             verticalalignment='center', horizontalalignment='left')
    fig.text(0.03, y_pos_row2, "Reconstructed", fontsize=12, rotation=0, 
             verticalalignment='center', horizontalalignment='left')

    # Adjust subplot parameters
    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1) # Adjusted spacing
    
    plt.show()
