import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from viT.dataset import VideoActionSegmentationDataset
from viT.model import VideoActionSegmentationModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json

from viT.losses import TotalLossWithContinuity
from utils.weight import smooth_class_weights
from utils.visualization import visualize_predictions, evaluate_and_plot_confusion_matrix

# Hyperparameters
DATA_DIR = 'dataset/rgb'
MEMFLOW_DIR = 'dataset/flow'
LABEL_DIR = 'dataset/anno_npy'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WINDOW_SIZE = 15
D_INPUT = 1024
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 4
USE_MEMFLOW = True
MASK = True
Weighted = True
Smooth_lambda = 0
lambda_continuity = 0
verbose = True
BINARY_CLASS = 0  # Set this to the class number for binary classification (i != 0)

hyperparameters = {
    'DATA_DIR': DATA_DIR,
    'MEMFLOW_DIR': MEMFLOW_DIR,
    'LABEL_DIR': LABEL_DIR,
    'BATCH_SIZE': BATCH_SIZE,
    'LEARNING_RATE': LEARNING_RATE,
    'NUM_EPOCHS': NUM_EPOCHS,
    'WINDOW_SIZE': WINDOW_SIZE,
    'D_INPUT': D_INPUT,
    'D_MODEL': D_MODEL,
    'NHEAD': NHEAD,
    'NUM_LAYERS': NUM_LAYERS,
    'USE_MEMFLOW': USE_MEMFLOW,
    'MASK': MASK,
    'Weighted': Weighted,
    'Smooth_lambda': Smooth_lambda,
    'lambda_continuity': lambda_continuity,
    'BINARY_CLASS': BINARY_CLASS
}

# Adjust NUM_CLASSES based on BINARY_CLASS
if BINARY_CLASS == 0:
    NUM_CLASSES = 17
else:
    NUM_CLASSES = 2

# Prepare datasets and data loaders
with open('dataset/data_info.json', 'r') as f:
    data_info = json.load(f)
video_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
train_files = [
    session + '.npy' for session in data_info['50salads']['train_session_set']]
val_files = [session + '.npy' for session in data_info['50salads']
             ['test_session_set']]

train_dataset = VideoActionSegmentationDataset(
    DATA_DIR, LABEL_DIR, MEMFLOW_DIR, train_files, window_size=WINDOW_SIZE, training=True)
val_dataset = VideoActionSegmentationDataset(
    DATA_DIR, LABEL_DIR, MEMFLOW_DIR, val_files, window_size=WINDOW_SIZE, training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model, Loss, and Optimizer
model = VideoActionSegmentationModel(d_input=D_INPUT, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                                     num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, use_memflow=USE_MEMFLOW, mask=MASK)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f'Using device: {device}')

if Weighted:
    if BINARY_CLASS != 0:
        class_counts = np.zeros(2)
        for _, _, labels in train_loader:
            labels_class = torch.argmax(labels, dim=2)
            labels_class = (labels_class == BINARY_CLASS).long().view(-1).cpu().numpy()
            for label in labels_class:
                class_counts[label] += 1
    else:
        class_counts = np.zeros(NUM_CLASSES)
        for _, _, labels in train_loader:
            labels_class = torch.argmax(labels, dim=2).view(-1).cpu().numpy()
            for label in labels_class:
                class_counts[label] += 1

    if Smooth_lambda == 0:
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / \
            np.sum(class_weights) * NUM_CLASSES
        class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    else:
        class_weights = smooth_class_weights(
            class_counts=class_counts, smoothing_factor=Smooth_lambda, device=device, verbose=verbose)
        
    if BINARY_CLASS != 0:
        class_weights = class_weights ** 2
    if verbose:
        print(class_counts)
        print(class_weights)
else:
    class_weights = None



if lambda_continuity == 0:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = TotalLossWithContinuity(
        lambda_continuity=lambda_continuity, num_classes=NUM_CLASSES, class_weights=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
best_val_accuracy = 0.0
best_model_state = None
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Create a unique folder for the entire training run based on hyperparameters
hyperparams_str = f"ws{WINDOW_SIZE}_dmodel{D_MODEL}_layers{NUM_LAYERS}_MASK{MASK}_Weighted_{Weighted}_SmoothLambda_{Smooth_lambda}_LambdaCont_{lambda_continuity}_Aug_BINARY_CLASS_{BINARY_CLASS}"
run_output_dir = os.path.join(output_dir, hyperparams_str)
os.makedirs(run_output_dir, exist_ok=True)

with open(os.path.join(run_output_dir, 'hyperparams.json'), 'w') as f:
    json.dump(hyperparameters, f, indent=4)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch") as pbar:
        for features, memflow, labels in train_loader:
            features, memflow, labels = features.to(
                device), memflow.to(device), labels.to(device)
            labels_class = torch.argmax(labels, dim=2)
            if BINARY_CLASS != 0:
                labels_class = (labels_class == BINARY_CLASS).long()

            # Forward pass
            outputs = model(features, memflow=memflow)

            # outputs(64, 15, NUM_CLASSES), labels_class(64, 15)
            if lambda_continuity == 0:
                loss = criterion(outputs.view(-1, NUM_CLASSES),
                                 labels_class.view(-1))
            else:
                loss = criterion(outputs, labels_class)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=2)
            if BINARY_CLASS != 0:
                predictions = (predictions == 1).long()
            correct_predictions += (predictions == labels_class).sum().item()
            total_predictions += labels_class.numel()
            pbar.update(1)

    train_accuracy = correct_predictions / total_predictions
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for features, memflow, labels in val_loader:
            features, memflow, labels = features.to(
                device), memflow.to(device), labels.to(device)
            labels_class = torch.argmax(labels, dim=2)
            if BINARY_CLASS != 0:
                labels_class = (labels_class == BINARY_CLASS).long()

            outputs = model(features, memflow=memflow)

            if lambda_continuity == 0:
                loss = criterion(outputs.view(-1, NUM_CLASSES),
                                 labels_class.view(-1))
            else:
                loss = criterion(outputs, labels_class)

            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=2)
            if BINARY_CLASS != 0:
                predictions = (predictions == 1).long()
            correct_predictions += (predictions == labels_class).sum().item()
            total_predictions += labels_class.numel()

    val_accuracy = correct_predictions / total_predictions
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict()
        torch.save(best_model_state, os.path.join(
            run_output_dir, 'best_model.pth'))
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}')
    print(torch.cuda.memory_allocated(device='cuda') /
          (1024 ** 2), "MB")

model.load_state_dict(best_model_state)

evaluate_and_plot_confusion_matrix(model, val_loader, run_output_dir, NUM_CLASSES=NUM_CLASSES, device=device)

visualize_predictions(model, os.path.join(DATA_DIR, 'SK047.npy'), os.path.join(
    MEMFLOW_DIR, 'SK047.npy'), os.path.join(LABEL_DIR, 'SK047.npy'), run_output_dir, device=device, WINDOW_SIZE=WINDOW_SIZE, NUM_CLASSES=NUM_CLASSES)
