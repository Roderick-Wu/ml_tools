import torch
import torch.nn as nn
import model
import data
from torch.utils.data import DataLoader, random_split


def load_model(model_path, device):
    loaded_model = model.simple_classifier_2(x_input_dim=640, hidden_dim=1024, output_dim=2)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


def run_inference_task(model, device, directory, task_list, batch_size=64):
    # X_input: torch.Tensor of shape [N, input_dim]
    X, X2, t, y = data.load_data_2(directory=directory, task_list=task_list, num_frames=7)
    X, X2, t, y = data.flatten_and_tensorize(list(zip(X, X2, t, y)))
    X, X2, t, y = X.to(device), X2.to(device), t.to(device), y.to(device)
    dataset = data.metaworld_dataset_2(X, X2, t, y)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = torch.empty((0, 2), device=device)  # output_dim is 2 for binary classification
    labels = torch.empty((0, 2), device=device)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch_X, batch_X2, batch_t, batch_y in loader:
            outputs = model(batch_X, batch_X2, batch_t)
            predictions = torch.cat((predictions, outputs), dim=0)
            labels = torch.cat((labels, batch_y), dim=0)
            loss = loss_fn(outputs, batch_y)
            loss += loss.item()
        loss = loss/len(loader)

    #import pdb; pdb.set_trace()
    accuracy = (predictions.argmax(dim=1).eq(labels.argmax(dim=1))).float().mean()

    print(f"Average Loss: {loss:.4f}")
    # print(f"Accuracy: {100 * accuracy:.2f}%")

    return predictions.cpu(), accuracy


def run_inference(model, device, new_data):
    """
    Run inference on new data using the trained model.
    
    Args:
        model: The trained model.
        device: The device to run the model on (CPU or GPU).
        new_data: A torch.Tensor of shape [N, input_dim] for inference.
    
    Returns:
        predictions: The model's predictions for the new data.
    """
    model.eval()
    with torch.no_grad():
        new_data = new_data.to(device)
        predictions = model(new_data)
        return predictions.argmax(dim=1)  # Return the class with the highest score



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model= load_model("history/model_3.pt", device=device)
    print("Model loaded successfully.")

    directory = "/home/wuroderi/metaworld_tasks/unet_embeddings"





    task_list = ["basketball-v3", "button-press-topdown-v3", "button-press-v3", "dial-turn-v3", "door-open-v3", "faucet-close-v3", "handle-press-v3"]
    preds, accuracy = run_inference_task(model, device, directory, task_list)

    print(f"Accuracy on Trained Tasks: {100 * accuracy:.2f}%")





    task_list = ["faucet-open-v3"]
    preds, accuracy = run_inference_task(model, device, directory, task_list)

    print(f"Accuracy on New Tasks: {100 * accuracy:.2f}%")
