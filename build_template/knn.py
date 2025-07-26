import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch

import data


DELTAS_AND_ACCURACIES = {
    'euclidean': [],
    'cosine': [],
    'mahalanobis': [],
    #'pca_kmeans': []
}

class EmbeddingFailureScorer:
    def __init__(self, Esucc, Efail, distance_type='euclidean', k=5):
        self.Esucc = Esucc
        self.Efail = Efail
        self.distance_type = distance_type
        self.k = k

        if distance_type in ['euclidean', 'cosine']:
            self.nn_succ = NearestNeighbors(n_neighbors=k, metric=distance_type)
            self.nn_fail = NearestNeighbors(n_neighbors=k, metric=distance_type)
            self.nn_succ.fit(self.Esucc)
            self.nn_fail.fit(self.Efail)

        elif distance_type == 'mahalanobis':
            self.cov_succ = EmpiricalCovariance().fit(self.Esucc)
            self.cov_fail = EmpiricalCovariance().fit(self.Efail)

    def _fit_mahalanobis(self):
        self.cov_succ = EmpiricalCovariance().fit(self.Esucc)
        self.cov_fail = EmpiricalCovariance().fit(self.Efail)

    def _fit_pca_kmeans(self):
        self.pca = PCA(n_components=10)
        self.kmeans = KMeans(n_clusters=2)
        all_embeddings = np.vstack([self.Esucc, self.Efail])
        reduced = self.pca.fit_transform(all_embeddings)
        self.kmeans.fit(reduced)

    def _knn_dist(self, e, E, metric):
        """Compute average distance to k nearest neighbors in E."""
        nn = NearestNeighbors(n_neighbors=self.k, metric=metric)
        nn.fit(E)
        dists, _ = nn.kneighbors([e])
        return np.mean(dists)

    def score(self, e):
        """
        e: np.ndarray of shape (D,)
            Test embedding vector
        Returns:
            scalar score s = d(e, Esucc) - d(e, Efail)
        """
        if self.distance_type in ['euclidean', 'cosine']:
            ds = self._knn_dist(e, self.Esucc, self.distance_type)
            df = self._knn_dist(e, self.Efail, self.distance_type)
            return ds - df

        elif self.distance_type == 'mahalanobis':
            ds = self.cov_succ.mahalanobis([e])[0]
            df = self.cov_fail.mahalanobis([e])[0]
            return ds - df

        elif self.distance_type == 'pca_kmeans':
            e_reduced = self.pca.transform([e])
            cluster_center_dist = np.linalg.norm(self.kmeans.cluster_centers_ - e_reduced, axis=1)
            # Heuristic: Distance to nearest cluster center is the score
            return cluster_center_dist.min()

        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

    def score_batch(self, E):
        """
        E: np.ndarray of shape (N, D)
        Returns:
            scores: np.ndarray of shape (N,)
        """
        if self.distance_type in ['euclidean', 'cosine']:
            ds, _ = self.nn_succ.kneighbors(E)
            df, _ = self.nn_fail.kneighbors(E)
            return np.mean(ds, axis=1) - np.mean(df, axis=1)

        elif self.distance_type == 'mahalanobis':
            ds = self.cov_succ.mahalanobis(E)
            df = self.cov_fail.mahalanobis(E)
            return ds - df

        else:
            raise NotImplementedError(f"Batched scoring not implemented for: {self.distance_type}")


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

def inference_embeddings(tensors, val_split, k, distance_type):

    # Load beforehand so that the same tensors are used for all distance types
    X, X2, t, y, X_val, X2_val, t_val, y_val = tensors

    X = X.mean(axis=(-1, -2))  # Average over spatial dimensions
    X_val = X_val.mean(axis=(-1, -2))  # Average over

    Esucc, Efail = split_success_and_failure(X, y)

    scorer = EmbeddingFailureScorer(Esucc, Efail, distance_type=distance_type, k=k)

    #import pdb; pdb.set_trace()
    scores = scorer.score_batch(X_val.cpu().numpy())

    average_success_score = np.mean(scores[y_val.cpu().numpy()[:, 0] == 1])
    average_failure_score = np.mean(scores[y_val.cpu().numpy()[:, 1] == 1])

    print(f"Average Success Score: {average_success_score:.4f}")
    print(f"Average Failure Score: {average_failure_score:.4f}")

    plot_score_distributions(scores, y_val, title=f"Score Distribution ({distance_type}, k={k})", plot_file=f"baseline_plots/score_distribution_{distance_type}_k{k}.png")


    # Compute accuracy at different thresholds
    scores = scorer.score_batch(X_val.cpu().numpy())
    y_val_np = y_val.cpu().numpy()

    avg_success_score = np.mean(scores[y_val_np[:, 0] == 1])
    avg_failure_score = np.mean(scores[y_val_np[:, 1] == 1])

    best_delta, best_acc = find_best_threshold(scores, y_val_np)
    print(f"Best Threshold (δ): {best_delta:.4f}")
    print(f"Validation Accuracy at δ: {best_acc:.4f}")
    plot_accuracy_vs_threshold(scores, y_val_np, title=f"Accuracy vs. Threshold ({distance_type}, k={k})", plot_file=f"baseline_plots/accuracy_vs_threshold_{distance_type}_k{k}.png")

    DELTAS_AND_ACCURACIES[distance_type].append((best_delta, best_acc))


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





task_list = ["basketball-v3", "button-press-topdown-v3", "button-press-v3", "dial-turn-v3", "door-open-v3", "faucet-close-v3", "faucet-open-v3", "handle-press-v3"]
directory = "/home/wuroderi/metaworld_tasks/unet_embeddings"


if __name__ == "__main__":
    tensors = load_tensors(directory, task_list, num_frames=7, val_split=0.2)
    for k in [1, 3, 5, 10]:
        for distance_type in ['euclidean', 'cosine', 'mahalanobis']:
            print(f"Running inference with k={k}, distance_type={distance_type} =======================")
            inference_embeddings(tensors, val_split=0.2, k=k, distance_type=distance_type)


    for distance_type, deltas_and_accuracies in DELTAS_AND_ACCURACIES.items():
        deltas, accuracies = zip(*deltas_and_accuracies)
        best_delta = deltas[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        print(f"Best Threshold for {distance_type}: {best_delta:.4f}, Accuracy: {best_accuracy:.4f}")