import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch_geometric
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split


class BirthAgeWithLaplacian(LightningDataModule):
    def __init__(self, n_laplacian, *, n_nodes = 87, batch_size=32, connectome_kind = "nws", split_ratio=0.7) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_laplacian = n_laplacian

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(range(len(self.df)), train_size=split_ratio, random_state=42)
        self._visualize_split()
        self.mean_birth_age_on_train_set = self.df["birth_age"].iloc[self.train_index].mean() # used later for standardisation
        self.node_pe = self._get_laplacian_pe(n_laplacian, n_nodes)

    def _get_df_and_connectomes(self, connectome_kind):

        df = pd.read_csv("combined.tsv", sep="\t")
        csv_filenames = os.listdir("connectomes-csv")

        has_connectome = []
        connectomes = []

        for i, row in df.iterrows():
            file_name = f"sub-{row['participant_id']}-ses-{row['session_id']}-{connectome_kind}.csv"
            if file_name in csv_filenames:
                has_connectome.append(i)
                connectomes.append(
                    pd.read_csv(os.path.join("connectomes-csv", file_name), header=None).to_numpy()
                )

        df = df.loc[has_connectome,:].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self):
        sns.kdeplot(self.df["birth_age"].iloc[self.train_index])
        sns.kdeplot(self.df["birth_age"].iloc[self.val_index])
        plt.title("Distribution of birth age accross train (blue) and validation (orange) sets")
        plt.savefig("split_plot.png")
        plt.clf()

    def _get_laplacian_pe(self, n_laplacian, n_nodes):
        with torch.no_grad():
            t = torch_geometric.transforms.AddLaplacianEigenvectorPE(k=n_laplacian)
            x = torch.eye(n_nodes, dtype=torch.float32)
            e = torch.tensor(
                [[i, j] for i in range(n_nodes) for j in range(n_nodes)],
                dtype=torch.long
            ).t().contiguous()
            data = torch_geometric.data.Data(x=x, edge_index=e)
            data = t(data)
            return data.laplacian_eigenvector_pe

    def train_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(self.connectomes[i,...], dtype=torch.float32) # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set, dtype=torch.float32)
                ) 
                for i in self.train_index
            ],
            batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False
        )
    
    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(self.connectomes[i,...], dtype=torch.float32) # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set, dtype=torch.float32)
                ) 
                for i in self.val_index
            ],
            batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False
        )
