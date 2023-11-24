import torch
import torch.nn as nn
import torch_geometric

class BaselineGCN(nn.Module):
    def __init__(self, input_channels=87, hidden_conv_channels=10, out_conv_channels=3, hidden_mlp_features=5):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_conv_channels = hidden_conv_channels
        self.out_conv_channels = out_conv_channels
        self.hidden_mlp_features = hidden_mlp_features

        self.conv1 = torch_geometric.nn.dense.DenseGraphConv(in_channels=input_channels, out_channels=hidden_conv_channels)
        self.conv2 = torch_geometric.nn.dense.DenseGraphConv(in_channels=hidden_conv_channels, out_channels=hidden_conv_channels)
        self.conv3 = torch_geometric.nn.dense.DenseGraphConv(in_channels=hidden_conv_channels, out_channels=out_conv_channels)
        self.linear1 = nn.Linear(in_features=(out_conv_channels*input_channels), out_features=hidden_mlp_features)
        self.linear2 = nn.Linear(in_features=hidden_mlp_features, out_features=1)

    def forward(self, x, adj):
        """x: (87, 87), adj: (87, 87)."""
        x = self.conv1(x, adj)
        x = torch.relu(x)
        x = self.conv2(x, adj)
        x = torch.relu(x)
        x = self.conv3(x, adj)
        x = torch.relu(x)
        x = torch.reshape(x, (-1, (self.out_conv_channels*self.input_channels)))
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x