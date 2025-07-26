import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch

from nflows.flows import Flow
from nflows.transforms import (
    CompositeTransform,
    ReversePermutation,
    AffineCouplingTransform,
    ActNorm
)
from nflows.distributions import StandardNormal
from nflows.nn.nets import MLP

class RNDDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        # Fixed target network
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Freeze the target network
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        error = F.mse_loss(pred_out, target_out, reduction='none').mean(dim=1)  # Per sample
        return error  # Higher = more OOD

    def train_step(self, x, optimizer):
        self.train()
        optimizer.zero_grad()
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        loss = F.mse_loss(pred_out, target_out)
        loss.backward()
        optimizer.step()
        return loss.item()



def train_rnd_model(X_np, num_epochs=100, batch_size=64, lr=1e-3, plot_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)

    model = RNDDetector(input_dim=X_np.shape[1], hidden_dim=X_np.shape[1]*2, output_dim=X_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in range(num_epochs):
        for batch in loader:
            loss = model.train_step(batch[0], optimizer)
            losses.append(loss)

        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

    # Plotting the training loss
    if plot_file:
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RND Predictor Training Loss')
        plt.legend()
        plt.savefig(plot_file)

    return model


def compute_rnd_scores(model_succ, model_fail, X_val):
    model_succ.eval()
    model_fail.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(next(model_succ.parameters()).device)
        s_succ = model_succ(X_tensor).cpu().numpy()
        s_fail = model_fail(X_tensor).cpu().numpy()
        return s_succ - s_fail  # Higher = more like failure


def split_success_and_failure(X, y):
    """
    Split data into successful and failed trajectories based on labels.
    
    Args:
        X: torch.Tensor of shape [N, D] - input data
        y: torch.Tensor of shape [N, 2] - labels (one-hot encoded)
    Returns:
        Esucc: np.ndarray of shape [N_succ, D] - successful embeddings
        Efail: np.ndarray of shape [N_fail, D] - failed embeddings
    """
    assert y.shape[1] == 2, "Labels must be one-hot encoded with 2 classes"
    
    # Convert to numpy for easier manipulation
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    Esucc = X_np[y_np[:, 0] == 1]  # Successful trajectories
    Efail = X_np[y_np[:, 1] == 1]  # Failed trajectories

    return Esucc, Efail

def load_tensors(directory, task_list, num_frames, val_split):
    X, X2, t, y = data.load_data_2(directory=directory, task_list=task_list, num_frames=num_frames)
    (X, X2, t, y), (X_val, X2_val, t_val, y_val) = data.shuffle_and_split_data(val_split, X, X2, t, y)

    return X, X2, t, y, X_val, X2_val, t_val, y_val

def inference_embeddings_rnd(tensors, val_split):

    # Load beforehand so that the same tensors are used for all distance types
    X, X2, t, y, X_val, X2_val, t_val, y_val = tensors

    X = X.mean(axis=(-1, -2))  # Average over spatial dimensions
    X_val = X_val.mean(axis=(-1, -2))  # Average over spatial dimensions

    # Convert to numpy and split
    Esucc, Efail = split_success_and_failure(X, y)
    X_val_np = X_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()

    # Train two RND models
    print("Training RND model for successful trajectories...")
    model_succ = train_rnd_model(Esucc, num_epochs=100, batch_size=64, lr=1e-3, plot_file="baseline_plots/rnd_success_training_loss.png")
    model_fail = train_rnd_model(Efail, num_epochs=100, batch_size=64, lr=1e-3, plot_file="baseline_plots/rnd_failure_training_loss.png")

    # Compute RND failure scores
    scores = compute_rnd_scores(model_succ, model_fail, X_val_np)

    # Evaluate with threshold selection
    best_delta, best_acc = find_best_threshold(scores, y_val_np)
    print(f"Best Threshold (δ): {best_delta:.4f}")
    print(f"Validation Accuracy at δ: {best_acc:.4f}")

    # Plotting
    plot_score_distributions(scores, y_val, title="RND Failure Score Distribution", plot_file="baseline_plots/rnd_score_distribution.png")
    plot_accuracy_vs_threshold(scores, y_val_np, title="RND Detector Accuracy vs Threshold",
                               plot_file="baseline_plots/rnd_accuracy_vs_threshold.png")


# Fix for nflows provided mlp, use this custom one for now
class SimpleContextMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes=[128, 128], activation=nn.ReLU):
        super().__init__()
        layers = []
        sizes = [in_features] + hidden_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())

        layers.append(nn.Linear(sizes[-1], out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x, context=None): # Ignore the context
        return self.net(x)


def create_logpzo_model(input_dim, hidden_dim=128, num_layers=5):
    def create_net(in_features, out_features):
        return SimpleContextMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_sizes=[hidden_dim]
        )

    transforms = []
    for _ in range(num_layers):
        transforms.append(ActNorm(features=input_dim))
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(
            AffineCouplingTransform(
                mask=torch.arange(input_dim) % 2,  # Alternating mask
                transform_net_create_fn=create_net,
            )
        )

    transform = CompositeTransform(transforms)
    base_dist = StandardNormal([input_dim])
    return Flow(transform, base_dist)

def train_logpzo_model(X_np, num_epochs=100, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)

    model = create_logpzo_model(input_dim=X_np.shape[1], hidden_dim=X_np.shape[1]*2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(X_tensor, batch_size=batch_size, shuffle=True)

    #import pdb; pdb.set_trace()
    model.train()
    for epoch in range(num_epochs):
        for batch in loader:
            loss = -model.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def compute_logpzo_scores(model_succ, model_fail, X_val):
    model_succ.eval()
    model_fail.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(next(model_succ.parameters()).device)
        logp_succ = model_succ.log_prob(X_tensor).cpu().numpy()
        logp_fail = model_fail.log_prob(X_tensor).cpu().numpy()
        return -logp_succ + logp_fail

def inference_embeddings_logpzo(tensors, val_split):
    # Load beforehand so that the same tensors are used for all distance types
    X, X2, t, y, X_val, X2_val, t_val, y_val = tensors

    X = X.mean(axis=(-1, -2))  # Average over spatial dimensions
    X_val = X_val.mean(axis=(-1, -2))  # Average over spatial dimensions

    # Convert to numpy and split
    Esucc, Efail = split_success_and_failure(X, y)
    X_val_np = X_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()

    # Train two logpzo models
    print("Training logpzo model for successful trajectories...")
    model_succ = train_logpzo_model(Esucc, num_epochs=100, batch_size=64, lr=1e-3)
    model_fail = train_logpzo_model(Efail, num_epochs=100, batch_size=64, lr=1e-3)

    # Compute logpzo failure scores
    scores = compute_logpzo_scores(model_succ, model_fail, X_val_np)

    # Evaluate with threshold selection
    best_delta, best_acc = find_best_threshold(scores, y_val_np)
    print(f"Best Threshold (δ): {best_delta:.4f}")
    print(f"Validation Accuracy at δ: {best_acc:.4f}")

    # Plotting
    plot_score_distributions(scores, y_val, title="logpzo Failure Score Distribution", plot_file="baseline_plots/logpzo_score_distribution.png")
    plot_accuracy_vs_threshold(scores, y_val_np, title="logpzo Detector Accuracy vs Threshold",
                               plot_file="baseline_plots/logpzo_accuracy_vs_threshold.png")



def plot_score_distributions(scores, y_val, title, plot_file):
    y_val_np = y_val.cpu().numpy()
    success_mask = y_val_np[:, 0] == 1
    failure_mask = y_val_np[:, 1] == 1

    plt.figure(figsize=(10, 6))
    plt.hist(scores[success_mask], bins=50, alpha=0.6, label='Success', color='green')
    plt.hist(scores[failure_mask], bins=50, alpha=0.6, label='Failure', color='red')
    plt.axvline(0, color='black', linestyle='--', label='Zero Threshold')
    plt.xlabel("Failure Score (d_succ - d_fail)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file)

    plt.close()

def plot_accuracy_vs_threshold(scores, y_true, title, plot_file):
    labels = y_true[:, 1]
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    accuracies = []

    for delta in thresholds:
        preds = (scores > delta).astype(int)
        acc = accuracy_score(labels, preds)
        accuracies.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.axvline(x=thresholds[np.argmax(accuracies)], color='red', linestyle='--', label=f'Best δ = {thresholds[np.argmax(accuracies)]:.4f}')
    plt.xlabel('Threshold δ')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file)

    plt.close()

def find_best_threshold(scores, y_true):
    """
    Finds the threshold delta that gives the best classification accuracy.

    Args:
        scores: np.ndarray of shape [N,] - failure scores
        y_true: np.ndarray of shape [N, 2] - one-hot labels (0=success, 1=failure)

    Returns:
        best_delta: float - threshold that maximizes accuracy
        best_accuracy: float - corresponding accuracy
    """
    # Ground truth: 1 = failure, 0 = success
    labels = y_true[:, 1]
    
    # Sort unique scores to use as candidate thresholds
    thresholds = np.unique(scores)
    best_acc = 0
    best_delta = 0

    for delta in thresholds:
        preds = (scores > delta).astype(int)  # Predict 1 (failure) if score > delta
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_delta = delta

    return best_delta, best_acc


if __name__ == "__main__":
    task_list = ["basketball-v3", "button-press-topdown-v3", "button-press-v3", "dial-turn-v3", "door-open-v3", "faucet-close-v3", "faucet-open-v3", "handle-press-v3"]
    directory = "/home/wuroderi/metaworld_tasks/unet_embeddings"
    num_frames = 7
    val_split = 0.2

    tensors = load_tensors(directory, task_list, num_frames, val_split)
    inference_embeddings_rnd(tensors, val_split)
    inference_embeddings_logpzo(tensors, val_split)