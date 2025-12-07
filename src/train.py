import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
import warnings
from sklearn.metrics import classification_report, confusion_matrix

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*ShiftScaleRotate.*")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_dataloaders
from src.model import get_model
import time
import copy
import os
import numpy as np
import mlflow
import mlflow.pytorch
from collections import Counter

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, device='cpu', patience=7):
    since = time.time()
    
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            all_preds = []
            all_labels = []
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Collect predictions for detailed report
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # MPS fix: use float() instead of double()
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Detailed report for validation
            if phase == 'val':
                target_names = ['NORMAL', 'PNEUMONIA', 'COVID-19']
                report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
                print("Detailed Classification Report:")
                print(classification_report(all_labels, all_preds, target_names=target_names))
                
                # Log detailed metrics to MLflow
                for class_name in target_names:
                    mlflow.log_metric(f"val_{class_name}_precision", report[class_name]['precision'], step=epoch)
                    mlflow.log_metric(f"val_{class_name}_recall", report[class_name]['recall'], step=epoch)
                    mlflow.log_metric(f"val_{class_name}_f1", report[class_name]['f1-score'], step=epoch)
            
            # MLflow logging
            mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{phase}_acc", epoch_acc.item(), step=epoch)
            
            if phase == 'val':
                # Learning rate scheduling
                scheduler.step(epoch_loss)
                
                # Early stopping check
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    model.load_state_dict(best_model_wts)
                    return model, val_acc_history
                
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == "__main__":
    # Update this path if needed
    DATA_DIR = "/Users/erenaskin/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
    COVID_DIR = "/Users/erenaskin/.cache/kagglehub/datasets/tawsifurrahman/covid19-radiography-database/versions/5"
    ADDITIONAL_DIR = "/Users/erenaskin/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Data"
    NIH_DIR = "/Users/erenaskin/.cache/kagglehub/datasets/nih-chest-xrays/sample/versions/4"
    BACHRR_DIR = "/Users/erenaskin/.cache/kagglehub/datasets/bachrr/covid-chest-xray/versions/4"
    
    # Check for MPS (Apple Silicon) or CUDA
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        
    print(f"Using device: {DEVICE}")
    
    # MLflow Setup
    mlflow.set_experiment("X-Ray Pneumonia & COVID Detection")
    mlflow.start_run()
    
    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.0001  # Lower learning rate for fine-tuning
    NUM_EPOCHS = 15 # Increased from 10
    NUM_CLASSES = 3
    PATIENCE = 5  # Increased from 3
    WEIGHT_DECAY = 1e-4  # L2 regularization
    
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("patience", PATIENCE)
    mlflow.log_param("weight_decay", WEIGHT_DECAY)
    mlflow.log_param("model", "ResNet50-FineTuned-Augmented")
    
    dataloaders, datasets = get_dataloaders(DATA_DIR, covid_dir=COVID_DIR, additional_dir=ADDITIONAL_DIR, nih_dir=NIH_DIR, bachrr_dir=BACHRR_DIR, batch_size=BATCH_SIZE)
    
    # Calculate class weights to handle imbalance
    train_labels = datasets['train'].labels
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    print("Class distribution in training set:", class_counts)
    
    # Formula: Total / (Num_Classes * Class_Count)
    # Classes: 0: Normal, 1: Pneumonia, 2: COVID-19
    class_weights = []
    for i in range(NUM_CLASSES):
        count = class_counts[i]
        weight = total_samples / (NUM_CLASSES * count)
        class_weights.append(weight)
        
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"Using class weights: {class_weights}")
    mlflow.log_param("class_weights", str(class_weights))

    model = get_model(num_classes=NUM_CLASSES, feature_extract=False)  # Fine-tune entire model
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Optimize all parameters with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    try:
        # Train with scheduler and early stopping
        model, _ = train_model(model, dataloaders, criterion, optimizer, scheduler, 
                               num_epochs=NUM_EPOCHS, device=DEVICE, patience=PATIENCE)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/pneumonia_covid_resnet50.pth")
        print("Model saved to models/pneumonia_covid_resnet50.pth")
        
        # Log model to MLflow
        # Ensure input_example is float32 to match model weights
        input_example = np.zeros((1, 3, 224, 224), dtype=np.float32)
        # Move model to CPU before logging to avoid device mismatch issues
        model_cpu = copy.deepcopy(model).to('cpu')
        mlflow.pytorch.log_model(model_cpu, name="model", input_example=input_example)
        
    finally:
        mlflow.end_run()
