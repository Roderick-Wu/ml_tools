import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Optional: Use this function if you don't already have one
def reduce_dimensionality(data, method, n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method {method}")
    return reducer.fit_transform(data)

def process_embeddings_multiple_tasks(file_list, reduction_method, output_file, color_labels):
    print(f"Processing {len(file_list)} files for {reduction_method.upper()} reduction...")
    print(f"Output will be saved to {output_file}")

    num_frames = 7
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    channels = 640

    video_embeddings = np.empty((0, channels))

    for task in file_list:
        for file_path in file_list[task]:
            data = np.load(file_path)  # shape: (B, C, 7, H, W)
            assert data.shape[2] == 7, "Expected 7 frames per video"

            averaged = data.mean(axis=(-1, -2))  # -> (B, C, 7)
            averaged = averaged.transpose(0, 2, 1)  # -> (B, 7, C)
            averaged = averaged[0] # -> (7, C) for single video


            for frame_idx in range(7):
                frame_embeddings = averaged[frame_idx, :]  # (1, C)
                video_embeddings = np.concatenate((video_embeddings, frame_embeddings.reshape(1, -1)), axis=0)
                # [ep0 f0, ep0 f1, ..., ep0 f6, ep1 f0, ep1 f1, ..., ep1 f6, ...]

    # import pdb; pdb.set_trace()
    # for frame in range(num_frames):
    #     video_embeddings_by_frame[frame] = [np.concatenate(embs, axis=0) for embs in video_embeddings_by_frame[frame]]

    reduced = reduce_dimensionality(video_embeddings, method=reduction_method)

    
    base_colors = ["blue", "green", "yellow", "orange", "red", "purple", "pink"]
    multi_cmap = LinearSegmentedColormap.from_list("multi_cmap", base_colors, N=len(file_list))
    colors = [multi_cmap(i / (len(file_list) - 1)) for i in range(len(file_list))]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(20, 14))

    for i, task in enumerate(file_list):
        print(f"Processing task number {i}: {task}")
        for frame_idx in range(7):
            for episode_num in range(len(file_list[task])):
                idx = i*len(file_list[task])*7 + episode_num*7 + frame_idx
                #print(f"Processing frame {frame_idx} for episode {episode_num} in task {task} at index {idx}")
                if color_labels[task][frame_idx][episode_num] == "blue":
                    ax.scatter(reduced[idx, 0], reduced[idx, 1], color=colors[i], s=5, marker='.')
                else:
                    ax.scatter(reduced[idx, 0], reduced[idx, 1], color=colors[i], s=5, marker='x')
                # ax.scatter(reduced[i*len(file_list[task])*7 + episode_num*7 + frame_idx, 0], reduced[i*len(file_list[task])*7 + episode_num*7 + frame_idx, 1], color=colors[i], s=5)


    bounds = np.linspace(0, 1, len(file_list) + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    tick_locs = (bounds[:-1] + bounds[1:]) / 2
    tick_labels = [task for task in file_list.keys()]

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        # cax=plt.axes([0.86, 0.15, 0.02, 0.7]), 
        ax=ax, 
        fraction=0.046, pad=0.04,
        ticks=tick_locs
    )
    cbar.ax.set_yticklabels(tick_labels)  # Blank y-tick labels
    #cbar.set_label('Success Frame Colors', rotation=270, labelpad=15)


    ax.set_title(f"{reduction_method.upper()} on Frame Embeddings")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)

    fig.tight_layout()

    fig.savefig(output_file, dpi=1200)

    plt.close(fig)  # Free memory

def process_embeddings_frames_together(file_list, reduction_method, output_file, color_labels):
    print(f"Processing {len(file_list)} files for {reduction_method.upper()} reduction...")
    print(f"Output will be saved to {output_file}")

    num_frames = 7
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    channels = 640

    video_embeddings_by_frame = np.empty((0, channels))

    for file_path in file_list:
        data = np.load(file_path)  # shape: (B, C, 7, H, W)
        assert data.shape[2] == 7, "Expected 7 frames per video"

        averaged = data.mean(axis=(-1, -2))  # -> (B, C, 7)
        averaged = averaged.transpose(0, 2, 1)  # -> (B, 7, C)
        averaged = averaged[0] # -> (7, C) for single video


        for frame_idx in range(7):
            frame_embeddings = averaged[frame_idx, :]  # (1, C)
            video_embeddings_by_frame = np.concatenate((video_embeddings_by_frame, frame_embeddings.reshape(1, -1)), axis=0)
            # [ep0 f0, ep0 f1, ..., ep0 f6, ep1 f0, ep1 f1, ..., ep1 f6, ...]

    reduced = reduce_dimensionality(video_embeddings_by_frame, method=reduction_method) # All embeddings for all episodes and frames together

    light_blue = np.array([0.7, 0.85, 1.0])  # light blue
    dark_blue  = np.array([0.0, 0.2, 0.6])   # dark blue

    light_red = np.array([1.0, 0.7, 0.7])   # light red
    dark_red  = np.array([0.6, 0.0, 0.0])   # dark red

    colors_success = np.linspace(light_blue, dark_blue, num_frames)
    colors_failure = np.linspace(light_red, dark_red, num_frames)

    cmap_success = ListedColormap(colors_success)
    cmap_failure = ListedColormap(colors_failure)

    fig, ax = plt.subplots(figsize=(10, 7))

    for frame_idx in range(7):
        for episode_num in range(len(file_list)):
            if color_labels[frame_idx][episode_num] == "blue":
                ax.scatter(reduced[episode_num*7 + frame_idx, 0], reduced[episode_num*7 + frame_idx, 1], color=colors_success[frame_idx], s=5)
            else:
                ax.scatter(reduced[episode_num*7 + frame_idx, 0], reduced[episode_num*7 + frame_idx, 1], color=colors_failure[frame_idx], s=5)

        #video_embeddings_by_frame[frame_idx].append(frame_embeddings.reshape(1, -1))  # Append as (1, C)

    bounds = np.linspace(0, 1, num_frames + 1)
    norm = BoundaryNorm(bounds, cmap_success.N)
    tick_locs = (bounds[:-1] + bounds[1:]) / 2
    tick_labels = [f"Frame {i}" for i in range(num_frames)]

    cbar_success = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_success, norm=norm),
        # cax=plt.axes([0.86, 0.15, 0.02, 0.7]), 
        ax=ax, 
        fraction=0.046, pad=0.04,
        ticks=tick_locs
    )
    cbar_success.ax.set_yticklabels(tick_labels)  # Blank y-tick labels
    #cbar_success.set_label('Success Frame Colors', rotation=270, labelpad=15)

    cbar_failure = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_failure, norm=norm),
        # cax=plt.axes([0.86, 0.15, 0.02, 0.7]), 
        ax=ax, 
        fraction=0.046, pad=0.04,
        ticks=tick_locs
    )
    cbar_failure.ax.set_yticklabels([" "] * num_frames)  # Set y-tick labels
    #cbar_failure.set_label('Failure Frame Colors', rotation=270, labelpad=15)


    ax.set_title(f"{reduction_method.upper()} on Frame Embeddings")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)

    fig.tight_layout()

    fig.savefig(output_file, dpi=300)

    plt.close(fig)  # Free memory

def process_embeddings_all(file_list, reduction_method, output_dir, color_labels):
    print(f"Processing {len(file_list)} files for {reduction_method.upper()} reduction...")
    print(f"Output will be saved to {output_dir}")

    num_frames = 7
    os.makedirs(output_dir, exist_ok=True)

    channels = 640

    video_embeddings_by_frame = [np.empty((0, channels)) for _ in range(num_frames)]

    for file_path in file_list:
        data = np.load(file_path)  # shape: (B, C, 7, H, W)
        assert data.shape[2] == 7, "Expected 7 frames per video"

        averaged = data.mean(axis=(-1, -2))  # -> (B, C, 7)
        averaged = averaged.transpose(0, 2, 1)  # -> (B, 7, C)
        averaged = averaged[0] # -> (7, C) for single video


        for frame_idx in range(7):
            frame_embeddings = averaged[frame_idx, :]  # (1, C)
            #video_embeddings_by_frame[frame_idx].append(frame_embeddings.reshape(1, -1))  # Append as (1, C)
            video_embeddings_by_frame[frame_idx] = np.concatenate((video_embeddings_by_frame[frame_idx], frame_embeddings.reshape(1, -1)), axis=0)

    # import pdb; pdb.set_trace()
    # for frame in range(num_frames):
    #     video_embeddings_by_frame[frame] = [np.concatenate(embs, axis=0) for embs in video_embeddings_by_frame[frame]]


    fig, axes = plt.subplots(1, 7, figsize=(42, 6))

    fig.suptitle(f"{reduction_method.upper()} on Frame Embeddings", fontsize=16)
    for frame in range(num_frames):
        reduced = reduce_dimensionality(video_embeddings_by_frame[frame], method=reduction_method)

        for i in range(len(file_list)):
            axes[frame].scatter(reduced[i, 0], reduced[i, 1], color=color_labels[frame][i], s=5)

        axes[frame].set_title(f"Frame {frame}")
        axes[frame].set_xlabel("Component 1")
        axes[frame].set_ylabel("Component 2")
        axes[frame].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output_dir, f"framewise_embeddings_{reduction_method}.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)  # Free memory
        print(f"Saved {reduction_method.upper()} plot to {save_path}")




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
                    frame_results[i-1].append("blue")
                else:
                    frame_results[i-1].append("red") # Failures

    return frame_results


#import pdb; pdb.set_trace()

#task = "basketball-v3"   
#task = "button-press-v3"
#task = "handle-press-v3"

tasks = ["basketball-v3", "button-press-topdown-v3", "button-press-v3", "dial-turn-v3", "door-open-v3", "faucet-close-v3", "faucet-open-v3", "handle-press-v3"]
reduction_method = "tsne"
#model_section = "end_input_blocks"
model_section = "start_output_blocks"

#import pdb; pdb.set_trace()
# diffusion_time_step = 0
for task in tasks:
    for diffusion_time_step in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        frame_colors = parse_results_csv(f"/home/wuroderi/metaworld_tasks/unet_embeddings/{task}/results.csv", num_frames=7)
        file_list = [f"/home/wuroderi/metaworld_tasks/unet_embeddings/{task}/episode_{i:03d}/{model_section}_{diffusion_time_step}.npy" for i in range(100)]

        process_embeddings_frames_together(file_list, reduction_method=reduction_method, output_file=f"/home/wuroderi/metaworld_tasks/unet_embeddings/PLOTTING/EVERYTHING/{task}/{model_section}_diffusion_time_step_{diffusion_time_step}.png", color_labels=frame_colors)



for diffusion_time_step in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    file_list = {}
    frame_colors = {}
    for task in tasks:
        frame_colors[task] = parse_results_csv(f"/home/wuroderi/metaworld_tasks/unet_embeddings/{task}/results.csv", num_frames=7)
        file_list[task] = [f"/home/wuroderi/metaworld_tasks/unet_embeddings/{task}/episode_{i:03d}/{model_section}_{diffusion_time_step}.npy" for i in range(100)]


    process_embeddings_multiple_tasks(file_list, reduction_method=reduction_method, output_file=f"/home/wuroderi/metaworld_tasks/unet_embeddings/PLOTTING/EVERYTHING/ALL_TASKS/{model_section}_diffusion_time_step_{diffusion_time_step}_{reduction_method}.png", color_labels=frame_colors)



