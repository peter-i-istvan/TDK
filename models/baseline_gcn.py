import torch
import torch.nn as nn
import torch_geometric


class BaselineGCN(nn.Module):
    def __init__(
        self,
        input_channels=87,
        hidden_conv_channels=10,
        out_conv_channels=3,
        hidden_mlp_features=5,
        final_sigmoid=False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_conv_channels = hidden_conv_channels
        self.out_conv_channels = out_conv_channels
        self.hidden_mlp_features = hidden_mlp_features

        self.conv1 = torch_geometric.nn.dense.DenseGraphConv(
            in_channels=input_channels, out_channels=hidden_conv_channels
        )
        self.prelu1 = torch.nn.PReLU()
        self.conv2 = torch_geometric.nn.dense.DenseGraphConv(
            in_channels=hidden_conv_channels, out_channels=hidden_conv_channels
        )
        self.prelu2 = torch.nn.PReLU()
        self.conv3 = torch_geometric.nn.dense.DenseGraphConv(
            in_channels=hidden_conv_channels, out_channels=out_conv_channels
        )
        self.prelu3 = torch.nn.PReLU()
        self.linear1 = nn.Linear(
            in_features=(out_conv_channels * 87), out_features=hidden_mlp_features
        )
        self.prelu4 = torch.nn.PReLU()
        self.linear2 = nn.Linear(in_features=hidden_mlp_features, out_features=1)
        if final_sigmoid:
            # probabilistic regression
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, adj):
        """x: (87, 87), adj: (87, 87)."""
        x = self.conv1(x, adj)
        x = self.prelu1(x)
        x = self.conv2(x, adj)
        x = self.prelu2(x)
        x = self.conv3(x, adj)
        x = self.prelu3(x)
        x = torch.reshape(x, (-1, (self.out_conv_channels * 87)))
        x = self.linear1(x)
        x = self.prelu4(x)
        x = self.linear2(x)
        if hasattr(self, "sigmoid"):
            x = self.sigmoid(x)
        return x
