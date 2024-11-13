import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix
from viT.model import VideoActionSegmentationModel

def visualize_predictions(model, features_path, memflow_path, label_path, output_dir, device='cuda', WINDOW_SIZE=30, NUM_CLASSES=17):
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

            window_predictions = torch.argmax(
                outputs, dim=2).cpu().numpy().flatten()
            predictions.extend(window_predictions)

    true_colors = labels.argmax(axis=1)

    plt.figure(figsize=(12, 5))  # Reduced height from 12 to 6
    plt.subplot(2, 1, 1)
    plt.imshow([predictions], aspect='auto',
               cmap='tab20', interpolation='nearest')
    plt.title('Predicted Classes')
    plt.yticks([])

    plt.subplot(2, 1, 2)
    plt.imshow([true_colors], aspect='auto',
               cmap='tab20', interpolation='nearest')
    plt.title('Ground Truth Classes')
    plt.yticks([])

    class_patches = [mpatches.Patch(color=plt.cm.tab20(
        i / NUM_CLASSES), label=f'Class {i}') for i in range(NUM_CLASSES)]
    plt.legend(handles=class_patches, loc='lower center', bbox_to_anchor=(0.5, -1.5),
               ncol=NUM_CLASSES // 2, frameon=False) 

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'predictions_vs_ground_truth.png'), bbox_inches='tight')
    plt.show()


def evaluate_and_plot_confusion_matrix(model, val_loader, output_dir, NUM_CLASSES=17, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, memflow, labels in val_loader:
            features, memflow, labels = features.to(
                device), memflow.to(device), labels.to(device)
            labels_class = torch.argmax(labels, dim=2)

            outputs = model(features, memflow=memflow)

            predictions = torch.argmax(outputs, dim=2)
            all_preds.extend(predictions.view(-1).cpu().numpy())
            all_labels.extend(labels_class.view(-1).cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(
        range(NUM_CLASSES)), normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=list(
        range(NUM_CLASSES)), yticklabels=list(range(NUM_CLASSES)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':


    show_confusion_matrix = True

    model_dir = 'output/bs64_lr0.0001_ws15_dmodel512_layers4_MASKTrue_Train_Reverse_LOG_50_Weight_CEL_Aug'
    best_model_state = torch.load(os.path.join(model_dir, 'best_model.pth'))
    hyperparameters = json.load(open(os.path.join(model_dir, 'hyperparams.json')))
    output_dir = os.path.join(model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VideoActionSegmentationModel(d_input=hyperparameters['D_INPUT'], d_model=hyperparameters['D_MODEL'], nhead=hyperparameters['NHEAD'], num_layers=hyperparameters['NUM_LAYERS'], num_classes=hyperparameters['NUM_CLASSES'], window_size=hyperparameters['WINDOW_SIZE'], use_memflow=hyperparameters['USE_MEMFLOW'], mask=hyperparameters['MASK']).to(device)
    model.load_state_dict(best_model_state)
    visualize_predictions(model, os.path.join(hyperparameters['DATA_DIR'], 'SK047.npy'), os.path.join(hyperparameters['MEMFLOW_DIR'], 'SK047.npy'), os.path.join(hyperparameters['LABEL_DIR'], 'SK047.npy'), output_dir, device=device, WINDOW_SIZE=hyperparameters['WINDOW_SIZE'], NUM_CLASSES=hyperparameters['NUM_CLASSES'])
