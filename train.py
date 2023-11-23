import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch_geometric
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import R2Score, PearsonCorrCoef

# https://pytorch.org/docs/stable/notes/randomness.html
# torch.manual_seed(42)
# use this instead:
# https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
seed_everything(42, workers=True)

BATCH_SIZE = 1

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch_geometric.nn.dense.DenseGraphConv(in_channels=87, out_channels=10)
        # self.bn1 = torch.nn.BatchNorm1d(87)
        self.conv2 = torch_geometric.nn.dense.DenseGraphConv(in_channels=10, out_channels=10)
        # self.bn2 = torch.nn.BatchNorm1d(87)
        self.conv3 = torch_geometric.nn.dense.DenseGraphConv(in_channels=10, out_channels=3)
        # self.bn3 = torch.nn.BatchNorm1d(87)
        self.linear1 = nn.Linear(in_features=3*87, out_features=5)
        self.linear2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x, adj):
        """x: (87, 87), adj: (87, 87)."""
        # omit batchnorm because it makes the training weak
        x = self.conv1(x, adj)
        x = torch.relu(x)
        # x = self.bn1(x)
        x = self.conv2(x, adj)
        x = torch.relu(x)
        # x = self.bn2(x)
        x = self.conv3(x, adj)
        x = torch.relu(x)
        # x = self.bn3(x)
        x = torch.reshape(x, (-1, 3*87))
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    
class Model(LightningModule):
    def __init__(self, model, loss_fn, optim) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim

        self.val_correlation = PearsonCorrCoef()
        self.val_r2 = R2Score()

    def training_step(self, batch, _):
        g, y_true = batch
        x, adj = g
        y_hat = self.model(x, adj).squeeze(1)

        loss = self.loss_fn(y_hat, y_true)
        self.log("Train MSE", loss)

        return loss
    
    def validation_step(self, batch):
        g, y_true = batch
        x, adj = g
        y_hat = self.model(x, adj).squeeze(1)

        loss = self.loss_fn(y_hat, y_true)
        self.log("Val MSE", loss)
        
        self.val_correlation.update(y_hat, y_true)
        self.val_r2.update(y_hat, y_true)
        
        return loss
    
    def on_validation_epoch_end(self):
        corr = self.val_correlation.compute()
        r2 = self.val_r2.compute()

        self.log_dict({"val_correlation": corr, "val_r2": r2})
    
    def configure_optimizers(self):
        return self.optim

def read_df_connectomes():
    df = pd.read_csv("combined.tsv", sep="\t")
    
    # read metadata and connectomes
    csv_filenames = os.listdir("connectomes-csv")

    has_connectome = []
    connectomes = []

    for i, row in df.iterrows():
        file_name = f"sub-{row['participant_id']}-ses-{row['session_id']}-nws.csv"
        if file_name in csv_filenames:
            has_connectome.append(i)
            connectomes.append(
                pd.read_csv(os.path.join("connectomes-csv", file_name), header=None).to_numpy()
            )

    df = df.loc[has_connectome,:].reset_index(drop=True)
    connectomes = np.array(connectomes)

    # outlier filtering
    mask = df["birth_weight"] > 0
    birth_weight_df = df[mask].copy().reset_index(drop=True)
    birth_weight_connectomes = connectomes[mask,...]

    return birth_weight_df, birth_weight_connectomes

def get_dataloaders(df, connectomes):
    split = int(673 * 0.7)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=[
            (
                (
                    torch.eye(87, dtype=torch.float32),
                    torch.tensor(connectomes[i,...], dtype=torch.float32)
                ),
                torch.tensor(df["birth_weight"][i], dtype=torch.float32)
            ) 
            for i in range(split)
        ],
        batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=[
            (
                (
                    torch.eye(87, dtype=torch.float32),
                    torch.tensor(connectomes[i,...], dtype=torch.float32)
                ),
                torch.tensor(df["birth_weight"][i], dtype=torch.float32)
            ) 
            for i in range(split, 673)
        ],
        batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True
    )

    return train_dataloader, test_dataloader

def main():
    df, connectomes = read_df_connectomes()
    train_dataloader, test_dataloader = get_dataloaders(df, connectomes)
    
    gcn = GCN()
    # this trainig works well with lower batch sizes
    # raise the batch size together with the learning rate
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = Model(gcn, loss_fn, optim)
    wandb_logger = WandbLogger(project="Onlab2")
    trainer = Trainer(
        max_epochs=20, log_every_n_steps=100, logger=wandb_logger, deterministic=True
    )
    trainer.fit(model, train_dataloader, test_dataloader)
    

if __name__ == "__main__":
    main()