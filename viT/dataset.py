import os
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoActionSegmentationDataset(Dataset):
    def __init__(self, data_dir, label_dir, memflow_dir, video_files, window_size=30, training=True):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.memflow_dir = memflow_dir
        self.video_files = video_files
        self.window_size = window_size
        self.training = training
        self.features_list, self.labels_list = self._load_all_windows()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_all_windows(self):
        features_list = []
        labels_list = []
        for video_file in self.video_files:
            data_path = os.path.join(self.data_dir, video_file)
            label_path = os.path.join(self.label_dir, video_file)
            memflow_path = os.path.join(self.memflow_dir, video_file)

            features = np.load(data_path)
            memflow = np.load(memflow_path)
            labels = np.load(label_path)

            T = features.shape[0]
            
            # Augumentation if training
            seed = np.random.randint(self.window_size) if self.training else 0
            
            for i in range(seed, T - self.window_size + 1, self.window_size):
                features_list.append((features[i:i + self.window_size], memflow[i:i + self.window_size]))
                labels_list.append(labels[i:i + self.window_size])

        return features_list, labels_list

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        features, memflow = self.features_list[idx]
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        memflow = torch.tensor(memflow, dtype=torch.float32).to(self.device)
        labels = torch.tensor(self.labels_list[idx], dtype=torch.float32).to(self.device)
        return features, memflow, labels


    

if __name__ == "__main__":
    
    data_dir = 'dataset/rgb'
    memflow_dir = 'dataset/flow'
    label_dir = 'dataset/anno_npy'

    video_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    dataset = VideoActionSegmentationDataset(data_dir, label_dir, memflow_dir, video_files, window_size=30)
    
    print("Dataset size:", len(dataset))

    sample_features, sample_memflow, sample_labels = dataset[0]
    print("Sample features size:", sample_features.size())
    print("Sample memflow size:", sample_memflow.size())
    print("Sample labels size:", sample_labels.size())