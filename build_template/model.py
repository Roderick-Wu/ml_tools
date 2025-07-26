import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.block(x)


class simple_classifier(nn.Module):
    def __init__(self, x_input_dim, hidden_dim, output_dim):
        super(simple_classifier, self).__init__()
        self.mlp = mlp(x_input_dim + 1, hidden_dim, output_dim)

    def forward(self, x, t):
        t_encoded = t.unsqueeze(-1)
        x = x.mean(axis=(-1, -2))  # Average over the last two dimensions (H, W)
        xt = torch.cat((x, t_encoded), dim=-1) # Shape: (B, C + 1)
        return nn.functional.softmax(self.mlp(xt), dim=-1)


class simple_classifier_2(nn.Module):
    def __init__(self, x_input_dim, hidden_dim, output_dim):
        super(simple_classifier_2, self).__init__()
        self.mlp = mlp(2*x_input_dim + 1, hidden_dim, output_dim)

    def forward(self, x, x2, t):
        t_encoded = t.unsqueeze(-1)
        x = x.mean(axis=(-1, -2))
        x2 = x2.mean(axis=(-1, -2))
        xt = torch.cat((x, x2, t_encoded), dim=-1) # Shape: (B, C + C + 1)
        return nn.functional.softmax(self.mlp(xt), dim=-1)
