import os
import numpy as np
import torch
from torch.utils.data import Dataset

class metaworld_dataset(Dataset):
    def __init__(self, X, t, y):
        self.X = X
        self.t = t
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.t[idx], self.y[idx]

class metaworld_dataset_2(Dataset):
    def __init__(self, X, X2, t, y):
        self.X = X
        self.X2 = X2
        self.t = t
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.X2[idx], self.t[idx], self.y[idx]


def parse_results_csv(file_path, num_frames=7):
    frame_results = [[] for _ in range(num_frames)]
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines[1:]:  # Skip header
            if line == "\n":
                continue
            
            parts = line.strip().split(',')
            frame = int(parts[0])

            for i in range(1, num_frames + 1):
                if parts[i]=="0":
                    frame_results[i-1].append(1)
                else:
                    frame_results[i-1].append(0) # Failures

    return frame_results

def load_data(directory, task_list, num_frames=7):
    X_list, t_list, y_list = [], [], []
    for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        task_files = {task: [f"{directory}/{task}/episode_{i:03d}/end_input_blocks_{t}.npy" for i in range(100)] for task in task_list}
        for task in task_files:
            raw_results = parse_results_csv(f"{directory}/{task}/results.csv", num_frames=7)

            for episode_num, file_path in enumerate(task_files[task]):
                tensor = np.load(file_path) # shape: (B, C, 7, H, W)
                tensor = torch.from_numpy(tensor)
                tensor = tensor[0, :, :, :, :]  # Batch size is always 1, now shape is (C, 7, H, W)
                for f in range(num_frames):
                    X_list.append(tensor[:, f, :, :])
                    t_list.append(t)
                    y_list.append(torch.tensor([1, 0] if raw_results[f][episode_num] == 1 else [0, 1], dtype=torch.float32))  # 1 for success, 0 for failure

    return X_list, t_list, y_list

# def load_data_2(directory, task_list, num_frames=7):
#     X_list, X2_list, t_list, y_list = [], [], [], []
#     for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
#         task_files = {task: [[f"{directory}/{task}/episode_{i:03d}/end_input_blocks_{t}.npy", f"{directory}/{task}/episode_{i:03d}/start_output_blocks_{t}.npy"] for i in range(100)] for task in task_list}
#         for task in task_files:
#             raw_results = parse_results_csv(f"{directory}/{task}/results.csv", num_frames=7)

#             for episode_num, (file_path, file_path2) in enumerate(task_files[task]):
#                 tensor = np.load(file_path) # shape: (B, C, 7, H, W)
#                 tensor = torch.from_numpy(tensor)
#                 tensor = tensor[0, :, :, :, :]  # Batch size is always 1, now shape is (C, 7, H, W)

#                 tensor2 = np.load(file_path2) # shape: (B, C, 7, H, W)
#                 tensor2 = torch.from_numpy(tensor2)
#                 tensor2 = tensor2[0, :, :, :, :]  # Batch size is always 1, now shape is (C, 7, H, W)

#                 for f in range(num_frames):
#                     X_list.append(tensor[:, f, :, :])
#                     X2_list.append(tensor2[:, f, :, :])
#                     t_list.append(t)
#                     y_list.append(torch.tensor([1, 0] if raw_results[f][episode_num] == 1 else [0, 1], dtype=torch.float32))  # 1 for success, 0 for failure

#     return X_list, X2_list, t_list, y_list


def load_data_2(directory, task_list, num_frames=7):
    X_list, X2_list, t_list, y_list = [], [], [], []
    for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        task_files = {task: [[f"{directory}/{task}/episode_{i:03d}/end_input_blocks_{t}.npy", f"{directory}/{task}/episode_{i:03d}/start_output_blocks_{t}.npy"] for i in range(100)] for task in task_list}
        for task in task_files:
            raw_results = parse_results_csv(f"{directory}/{task}/results.csv", num_frames=7)

            for episode_num, (file_path, file_path2) in enumerate(task_files[task]):
                tensor = np.load(file_path) # shape: (B, C, 7, H, W)
                tensor = torch.from_numpy(tensor)
                tensor = tensor[0, :, :, :, :]  # Batch size is always 1, now shape is (C, 7, H, W)

                tensor2 = np.load(file_path2) # shape: (B, C, 7, H, W)
                tensor2 = torch.from_numpy(tensor2)
                tensor2 = tensor2[0, :, :, :, :]  # Batch size is always 1, now shape is (C, 7, H, W)
                
                episode_X, episode_X2, episode_t, episode_y = [], [], [], []

                for f in range(num_frames):
                    episode_X.append(tensor[:, f, :, :])
                    episode_X2.append(tensor2[:, f, :, :])
                    episode_t.append(torch.tensor(t, dtype=torch.float32))
                    episode_y.append(torch.tensor([1, 0] if raw_results[f][episode_num] == 1 else [0, 1], dtype=torch.float32))  # 1 for success, 0 for failure

                X_list.append(episode_X)
                X2_list.append(episode_X2)
                t_list.append(episode_t)
                y_list.append(episode_y)


    # list[episodes][frame_number]

    return X_list, X2_list, t_list, y_list


def flatten_and_tensorize(trajectories):
    num_lists = len(trajectories[0])
    result = [[] for _ in range(num_lists)]
    for traj in trajectories: # traj = [X, X2, t, y]
        for i, frames in enumerate(traj):
            result[i].extend(frames)
    return [torch.stack(result[i]) for i in range(num_lists)]


def shuffle_and_split_data(val_split, *lists):
    l = list(zip(*lists)) # [(X, X2, t, y), ...]
    np.random.shuffle(l)
    split_index = int(len(lists[0]) * (1 - val_split))

    train_data = l[:split_index]
    val_data = l[split_index:]

    train_data = flatten_and_tensorize(train_data)
    val_data = flatten_and_tensorize(val_data)

    return train_data, val_data
