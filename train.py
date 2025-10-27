import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from resnet import model, optimizer, loss_fn
from prepare_data import train_loader, test_loader, device

NUM_EPOCHS = 20

# Early stopping parameters
PATIENCE = 5  # Number of epochs to wait before stopping
best_val_loss = float('inf')
epochs_without_improvement = 0
best_epoch = 0

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Create directory for saving models
os.makedirs('saved_models', exist_ok=True)

print("--- Starting Model Training ---")
print(f"Early stopping enabled with patience={PATIENCE}")
print(f"Best model will be saved to 'saved_models/best_model.pth'\n")

for epoch in range(NUM_EPOCHS):

    # --- Training Phase --- #
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Wrap train_loader with tqdm for progress bar
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]', 
                      leave=False, ncols=100)
    
    for inputs, labels in train_pbar:
        
        # Move data to the GPU (if available)
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(inputs)

        # 3. Calculate loss
        loss = loss_fn(outputs, labels.unsqueeze(1))

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

        # --- Collect stats --- # 
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels.unsqueeze(1)).sum().item()
        
        # Update progress bar with current loss
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate average stats for the epoch
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # --- Testing Phase --- #
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    # Wrap test_loader with tqdm for progress bar
    val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ', 
                    leave=False, ncols=100)
    
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels.unsqueeze(1))

            running_val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels.unsqueeze(1)).sum().item()
            
            # Update progress bar with current loss
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_val_loss = running_val_loss / len(test_loader)
    epoch_val_acc = correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    # Print epoch summary with colors
    print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    # --- Early Stopping & Model Checkpointing --- #
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        
        # Save the best model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_acc': epoch_train_acc,
            'val_acc': epoch_val_acc,
        }, 'saved_models/best_model.pth')
        
        print(f"  ✓ New best model saved! (Val Loss: {epoch_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"  ⚠ No improvement for {epochs_without_improvement} epoch(s)")
        
        if epochs_without_improvement >= PATIENCE:
            print(f"\n⏹ Early stopping triggered after {epoch + 1} epochs")
            print(f"Best model was at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
            break

print("--- Training Complete ---")
print(f"Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

# --- Load Best Model for Final Evaluation --- #
print("\n--- Loading Best Model for Final Evaluation ---")
checkpoint = torch.load('saved_models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best Val Loss: {checkpoint['val_loss']:.4f} | Best Val Acc: {checkpoint['val_acc']:.4f}\n")


plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


