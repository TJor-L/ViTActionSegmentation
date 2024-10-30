import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from viT.dataset import VideoActionSegmentationDataset
from viT.model import VideoActionSegmentationModel
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches
import json


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        CE_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return focal_loss.mean()

class MEFullLoss(nn.Module):
    def __init__(self, lambda_entropy=0.1):
        super(MEFullLoss, self).__init__()
        self.lambda_entropy = lambda_entropy
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Standard Cross Entropy Loss
        CE_loss = self.cross_entropy(logits, targets)
        
        # Compute entropy of the predictions
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        log_probs = torch.log(probs + 1e-6)  # Avoid log(0) issues
        entropy_loss = -torch.sum(probs * log_probs, dim=-1).mean()
        
        # Combine CE loss with entropy maximization term
        total_loss = CE_loss - self.lambda_entropy * entropy_loss
        
        return total_loss


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
NUM_CLASSES = 17
FocalLoss_alpha = 0.25
USE_MEMFLOW = True
MASK = True

# Prepare datasets and data loaders

with open('dataset/data_info.json', 'r') as f:
    data_info = json.load(f)
video_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
train_files = [session + '.npy' for session in data_info['50salads']['train_session_set']]
val_files = [session + '.npy' for session in data_info['50salads']['test_session_set']]

train_dataset = VideoActionSegmentationDataset(DATA_DIR, LABEL_DIR, MEMFLOW_DIR, train_files, window_size=WINDOW_SIZE, training=True)
val_dataset = VideoActionSegmentationDataset(DATA_DIR, LABEL_DIR, MEMFLOW_DIR, val_files, window_size=WINDOW_SIZE, training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, and Optimizer
model = VideoActionSegmentationModel(d_input=D_INPUT, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, 
                                     num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, use_memflow=USE_MEMFLOW, mask=MASK)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f'Using device: {device}')


# Calculate class weights based on the number of samples per class
class_counts = np.zeros(NUM_CLASSES)
for _, _, labels in train_loader:
    labels_class = torch.argmax(labels, dim=2).view(-1).cpu().numpy()
    for label in labels_class:
        class_counts[label] += 1

class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
class_weights = class_weights / np.sum(class_weights) * NUM_CLASSES  # Normalize weights
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
# weights = [1.235, 1.492, 1.493, 2.326, 2.418, 2.272, 1.961, 1.538, 2.439, 2.070, 2.081, 1.163, 2.027, 1.190, 2.000, 1.694, 2.500]
# class_weights = torch.tensor(weights, dtype=torch.float, device=device)
print(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
best_val_accuracy = 0.0
best_model_state = None
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Create a unique folder for the entire training run based on hyperparameters
hyperparams_str = f"bs{BATCH_SIZE}_lr{LEARNING_RATE}_ws{WINDOW_SIZE}_dmodel{D_MODEL}_layers{NUM_LAYERS}_MASK{MASK}_Train_Reverse_Weight_CEL_Aug"
run_output_dir = os.path.join(output_dir, hyperparams_str)
os.makedirs(run_output_dir, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for features, memflow, labels in train_loader:
        features, memflow, labels = features.to(device), memflow.to(device), labels.to(device)
        labels_class = torch.argmax(labels, dim=2)
        
        # Forward pass
        outputs = model(features, memflow=memflow)

        loss = criterion(outputs.view(-1, NUM_CLASSES), labels_class.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=2)
        correct_predictions += (predictions == labels_class).sum().item()
        total_predictions += labels_class.numel()

    train_accuracy = correct_predictions / total_predictions
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for features, memflow, labels in val_loader:
            features, memflow, labels = features.to(device), memflow.to(device), labels.to(device)
            labels_class = torch.argmax(labels, dim=2)
            
            outputs = model(features, memflow=memflow)

            loss = criterion(outputs.view(-1, NUM_CLASSES), labels_class.view(-1))
            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=2)
            correct_predictions += (predictions == labels_class).sum().item()
            total_predictions += labels_class.numel()

    val_accuracy = correct_predictions / total_predictions
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict()
        torch.save(best_model_state, os.path.join(run_output_dir, 'best_model.pth'))
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}')
    print(torch.cuda.memory_allocated(device='cuda') / (1024 ** 2), "MB")  # Output GPU memory usage

# Load best model
model.load_state_dict(best_model_state)

# Evaluate best model and generate confusion matrix
def evaluate_and_plot_confusion_matrix(model, val_loader, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, memflow, labels in val_loader:
            features, memflow, labels = features.to(device), memflow.to(device), labels.to(device)
            labels_class = torch.argmax(labels, dim=2)
            
            outputs = model(features, memflow=memflow)

            predictions = torch.argmax(outputs, dim=2)
            all_preds.extend(predictions.view(-1).cpu().numpy())
            all_labels.extend(labels_class.view(-1).cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)), normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=list(range(NUM_CLASSES)), yticklabels=list(range(NUM_CLASSES)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.show()

evaluate_and_plot_confusion_matrix(model, val_loader, run_output_dir)

# Visualization moved to separate code
def visualize_predictions(model, features_path, memflow_path, label_path, output_dir):
    model.eval()
    features = np.load(features_path)
    memflow = np.load(memflow_path)
    labels = np.load(label_path)

    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    memflow_tensor = torch.tensor(memflow, dtype=torch.float32).to(device)

    predictions = []
    with torch.no_grad():
        for i in range(0, features.shape[0] - WINDOW_SIZE + 1, WINDOW_SIZE):
            window_features = features_tensor[i:i + WINDOW_SIZE].unsqueeze(0)
            window_memflow = memflow_tensor[i:i + WINDOW_SIZE].unsqueeze(0)
            
            outputs = model(window_features, window_memflow)
            
            window_predictions = torch.argmax(outputs, dim=2).cpu().numpy().flatten()
            predictions.extend(window_predictions)

    true_colors = labels.argmax(axis=1)
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.imshow([predictions], aspect='auto', cmap='tab20', interpolation='nearest')
    plt.title('Predicted Classes')
    plt.yticks([])

    plt.subplot(2, 1, 2)
    plt.imshow([true_colors], aspect='auto', cmap='tab20', interpolation='nearest')
    plt.title('Ground Truth Classes')
    plt.yticks([])
    plt.xlabel('Frame Index')
    
    # Add a legend for class colors
    class_patches = [mpatches.Patch(color=plt.cm.tab20(i / NUM_CLASSES), label=f'Class {i}') for i in range(NUM_CLASSES)]
    plt.legend(handles=class_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_ground_truth.png'))
    plt.show()

visualize_predictions(model, os.path.join(DATA_DIR, 'SK047.npy'), os.path.join(MEMFLOW_DIR, 'SK047.npy'), os.path.join(LABEL_DIR, 'SK047.npy'), run_output_dir)
