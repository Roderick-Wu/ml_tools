import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
import model
import matplotlib.pyplot as plt
import data


def train(model, epochs, batch_size, gamma, device, val_split, directory, task_list, plot_file="loss_curve.png", model_save_path="simple_model.pt"):
    model = model.to(device)

    # Load data  
    #import pdb; pdb.set_trace()
    X, X2, t, y = data.load_data_2(directory=directory, task_list=task_list, num_frames=7)


    (X, X2, t, y), (X_val, X2_val, t_val, y_val) = data.shuffle_and_split_data(val_split, X, X2, t, y)

    #X, X2, t, y = torch.stack(X), torch.stack(X2), torch.tensor(t, dtype=torch.float32), torch.stack(y)
    X, X2, t, y = X.to(device), X2.to(device), t.to(device), y.to(device)
    train_dataset = data.metaworld_dataset_2(X, X2, t, y)

    #X_val, X2_val, t_val, y_val = torch.stack(X_val), torch.stack(X2_val), torch.tensor(t_val, dtype=torch.float32), torch.stack(y_val)
    X_val, X2_val, t_val, y_val = X_val.to(device), X2_val.to(device), t_val.to(device), y_val.to(device)
    val_dataset = data.metaworld_dataset_2(X_val, X2_val, t_val, y_val)

    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=gamma)

    #import pdb; pdb.set_trace()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        predictions = torch.empty((0, 2), device=device)
        labels = torch.empty((0, 2), device=device)
        total_loss = 0

        for batch_X, batch_X2, batch_t, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, batch_X2, batch_t)
            predictions = torch.cat((predictions, outputs), dim=0)
            labels = torch.cat((labels, batch_y), dim=0)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))
        train_accuracy = (predictions.argmax(dim=1).eq(labels.argmax(dim=1))).float().mean()
        train_accuracies.append(train_accuracy.item())

        # Validation step
        predictions = torch.empty((0, 2), device=device)
        labels = torch.empty((0, 2), device=device)
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_X, batch_X2, batch_t, batch_y in val_loader:
                outputs = model(batch_X, batch_X2, batch_t)
                predictions = torch.cat((predictions, outputs), dim=0)
                labels = torch.cat((labels, batch_y), dim=0)
                loss = loss_fn(outputs, batch_y)
                val_loss += loss.item()

            val_losses.append(val_loss/len(val_loader))
            val_accuracy = (predictions.argmax(dim=1).eq(labels.argmax(dim=1))).float().mean()
            val_accuracies.append(val_accuracy.item())

        print(f"Epoch {epoch+1}/{epochs} ==========================")
        print(f"Average Train Loss: {total_loss/len(train_loader):.4f}, Average Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Train Accuracy: {100 * train_accuracy:.2f}%, Validation Accuracy: {100 * val_accuracy:.2f}%")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_title("Training and Validation Losses")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.plot(train_losses, label='Train Loss', color='tab:blue')
    ax1.plot(val_losses, label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='tab:red')
    ax2.plot(train_accuracies, label='Train Accuracy', color='tab:green', linestyle='--')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='tab:red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(plot_file)
    print(f"Saved loss and accuracy plot to {plot_file}")

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.simple_classifier_2(x_input_dim=640, hidden_dim=1024, output_dim=2)

    plot_file = "history/loss_curve_basketball.png"
    model_save_path = "history/model_basketball.pt"
    epochs, batch_size, gamma, val_split = 100, 64, 0.0001, 0.2
    directory = "/home/wuroderi/metaworld_tasks/unet_embeddings"
    # task_list = ["basketball-v3", "button-press-topdown-v3", "button-press-v3", "dial-turn-v3", "door-open-v3", "faucet-close-v3", "handle-press-v3"]
    task_list = ["basketball-v3"]
    train(model, epochs=epochs, batch_size=batch_size, gamma=gamma, device=device, val_split=val_split, directory=directory, task_list=task_list, plot_file=plot_file, model_save_path=model_save_path)

    print(f"Training complete. Check {plot_file} and {model_save_path} for results.")