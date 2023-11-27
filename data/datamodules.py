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
    def __init__(
        self,
        n_laplacian,
        *,
        n_nodes=87,
        batch_size=32,
        connectome_kind="nws",
        split_ratio=0.7,
        split_plot_path=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_laplacian = n_laplacian

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(
            range(len(self.df)), train_size=split_ratio, random_state=42
        )
        if split_plot_path is not None:
            self._visualize_split(split_plot_path)
        self.mean_birth_age_on_train_set = (
            self.df["birth_age"].iloc[self.train_index].mean()
        )  # used later for standardisation
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
                    pd.read_csv(
                        os.path.join("connectomes-csv", file_name), header=None
                    ).to_numpy()
                )

        df = df.loc[has_connectome, :].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self, split_plot_path):
        sns.kdeplot(
            self.df["birth_age"].iloc[self.train_index], label="train", fill=True
        )
        sns.kdeplot(
            self.df["birth_age"].iloc[self.val_index], label="validation", fill=True
        )
        plt.title(
            f"Distribution of birth age across train (n={len(self.train_index)}) and validation (n={len(self.val_index)}) sets"
        )
        plt.legend()
        plt.savefig(split_plot_path)
        plt.clf()

    def _get_laplacian_pe(self, n_laplacian, n_nodes):
        with torch.no_grad():
            t = torch_geometric.transforms.AddLaplacianEigenvectorPE(k=n_laplacian)
            x = torch.eye(n_nodes, dtype=torch.float32)
            e = (
                torch.tensor(
                    [[i, j] for i in range(n_nodes) for j in range(n_nodes)],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )
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
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.train_index
            ],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.val_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )


class BirthAgeTrainValTestWithOneHot(LightningDataModule):
    def __init__(
        self,
        *,
        n_nodes=87,
        batch_size=32,
        connectome_kind="nws",
        train_ratio=0.7,
        val_to_test_ratio=0.5,
        split_plot_path=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(
            range(len(self.df)), train_size=train_ratio, random_state=42
        )
        self.val_index, self.test_index = train_test_split(
            self.val_index, train_size=val_to_test_ratio, random_state=42
        )
        if split_plot_path is not None:
            self._visualize_split(split_plot_path)
        # mean_birth_age_on_train_set used later for standardisation
        self.mean_birth_age_on_train_set = (
            self.df["birth_age"].iloc[self.train_index].mean()
        )
        self.n_nodes = n_nodes
        self.node_pe = torch.eye(n_nodes)

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
                    pd.read_csv(
                        os.path.join("connectomes-csv", file_name), header=None
                    ).to_numpy()
                )

        df = df.loc[has_connectome, :].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self, split_plot_path):
        sns.kdeplot(
            self.df["birth_age"].iloc[self.train_index], label="train", fill=True
        )
        sns.kdeplot(
            self.df["birth_age"].iloc[self.val_index], label="validation", fill=True
        )
        sns.kdeplot(self.df["birth_age"].iloc[self.test_index], label="test", fill=True)
        plt.title(
            f"Distribution of birth age across train (n={len(self.train_index)}), validation (n={len(self.val_index)})\n and test (n={len(self.test_index)}) sets"
        )
        plt.legend()
        plt.savefig(split_plot_path)
        plt.clf()

    def train_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.eye(self.n_nodes),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.train_index
            ],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.eye(self.n_nodes),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.val_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.eye(self.n_nodes),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.test_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )


class BirthAgeTrainValTestWithLaplacian(LightningDataModule):
    def __init__(
        self,
        n_laplacian,
        *,
        n_nodes=87,
        batch_size=32,
        connectome_kind="nws",
        train_ratio=0.7,
        val_to_test_ratio=0.5,
        split_plot_path=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_laplacian = n_laplacian

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(
            range(len(self.df)), train_size=train_ratio, random_state=42
        )
        self.val_index, self.test_index = train_test_split(
            self.val_index, train_size=val_to_test_ratio, random_state=42
        )
        if split_plot_path is not None:
            self._visualize_split(split_plot_path)
        # mean_birth_age_on_train_set used later for standardisation
        self.mean_birth_age_on_train_set = (
            self.df["birth_age"].iloc[self.train_index].mean()
        )
        # node_pe is (n_nodes, F) where F is the feature dimensionality
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
                    pd.read_csv(
                        os.path.join("connectomes-csv", file_name), header=None
                    ).to_numpy()
                )

        df = df.loc[has_connectome, :].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self, split_plot_path):
        sns.kdeplot(
            self.df["birth_age"].iloc[self.train_index], label="train", fill=True
        )
        sns.kdeplot(
            self.df["birth_age"].iloc[self.val_index], label="validation", fill=True
        )
        sns.kdeplot(self.df["birth_age"].iloc[self.test_index], label="test", fill=True)
        plt.title(
            f"Distribution of birth age across train (n={len(self.train_index)}), validation (n={len(self.val_index)})\n and test (n={len(self.test_index)}) sets"
        )
        plt.legend()
        plt.savefig(split_plot_path)
        plt.clf()

    def _get_laplacian_pe(self, n_laplacian, n_nodes):
        with torch.no_grad():
            t = torch_geometric.transforms.AddLaplacianEigenvectorPE(k=n_laplacian)
            x = torch.eye(n_nodes, dtype=torch.float32)
            e = (
                torch.tensor(
                    [[i, j] for i in range(n_nodes) for j in range(n_nodes)],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )
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
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.train_index
            ],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.val_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["birth_age"].iloc[i] - self.mean_birth_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.test_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )


class ScanAgeTrainValTestWithLaplacian(LightningDataModule):
    def __init__(
        self,
        n_laplacian,
        *,
        n_nodes=87,
        batch_size=32,
        connectome_kind="nws",
        train_ratio=0.7,
        val_to_test_ratio=0.5,
        split_plot_path=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_laplacian = n_laplacian

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(
            range(len(self.df)), train_size=train_ratio, random_state=42
        )
        self.val_index, self.test_index = train_test_split(
            self.val_index, train_size=val_to_test_ratio, random_state=42
        )
        if split_plot_path is not None:
            self._visualize_split(split_plot_path)
        # mean_birth_age_on_train_set used later for standardisation
        self.mean_scan_age_on_train_set = (
            self.df["scan_age"].iloc[self.train_index].mean()
        )
        # node_pe is (n_nodes, F) where F is the feature dimensionality
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
                    pd.read_csv(
                        os.path.join("connectomes-csv", file_name), header=None
                    ).to_numpy()
                )

        df = df.loc[has_connectome, :].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self, split_plot_path):
        sns.kdeplot(
            self.df["scan_age"].iloc[self.train_index], label="train", fill=True
        )
        sns.kdeplot(
            self.df["scan_age"].iloc[self.val_index], label="validation", fill=True
        )
        sns.kdeplot(self.df["scan_age"].iloc[self.test_index], label="test", fill=True)
        plt.title(
            f"Distribution of scan age across train (n={len(self.train_index)}), validation (n={len(self.val_index)})\n and test (n={len(self.test_index)}) sets"
        )
        plt.legend()
        plt.savefig(split_plot_path)
        plt.clf()

    def _get_laplacian_pe(self, n_laplacian, n_nodes):
        with torch.no_grad():
            t = torch_geometric.transforms.AddLaplacianEigenvectorPE(k=n_laplacian)
            x = torch.eye(n_nodes, dtype=torch.float32)
            e = (
                torch.tensor(
                    [[i, j] for i in range(n_nodes) for j in range(n_nodes)],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )
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
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.train_index
            ],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.val_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.test_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )


class ScanAgeTrainValTestWithLocalDegreeProfile(LightningDataModule):
    def __init__(
        self,
        *,
        n_nodes=87,
        batch_size=32,
        connectome_kind="nws",
        train_ratio=0.7,
        val_to_test_ratio=0.5,
        split_plot_path=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(
            range(len(self.df)), train_size=train_ratio, random_state=42
        )
        self.val_index, self.test_index = train_test_split(
            self.val_index, train_size=val_to_test_ratio, random_state=42
        )
        if split_plot_path is not None:
            self._visualize_split(split_plot_path)
        # mean_birth_age_on_train_set used later for standardisation
        self.mean_scan_age_on_train_set = (
            self.df["scan_age"].iloc[self.train_index].mean()
        )
        # node_pe is (n_nodes, F) where F is the feature dimensionality
        self.node_pe = self._get_local_degree_profile(n_nodes)

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
                    pd.read_csv(
                        os.path.join("connectomes-csv", file_name), header=None
                    ).to_numpy()
                )

        df = df.loc[has_connectome, :].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self, split_plot_path):
        sns.kdeplot(
            self.df["scan_age"].iloc[self.train_index], label="train", fill=True
        )
        sns.kdeplot(
            self.df["scan_age"].iloc[self.val_index], label="validation", fill=True
        )
        sns.kdeplot(self.df["scan_age"].iloc[self.test_index], label="test", fill=True)
        plt.title(
            f"Distribution of scan age across train (n={len(self.train_index)}), validation (n={len(self.val_index)})\n and test (n={len(self.test_index)}) sets"
        )
        plt.legend()
        plt.savefig(split_plot_path)
        plt.clf()

    def _get_local_degree_profile(self, n_nodes):
        with torch.no_grad():
            t = torch_geometric.transforms.LocalDegreeProfile()
            x = None
            e = (
                torch.tensor(
                    [[i, j] for i in range(n_nodes) for j in range(n_nodes)],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )
            data = torch_geometric.data.Data(x=x, edge_index=e)
            data = t(data)
            return data.x

    def train_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.train_index
            ],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.val_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.test_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )


class ScanAgeTrainValTestWithOnes(LightningDataModule):
    def __init__(
        self,
        n_ones,
        *,
        n_nodes=87,
        batch_size=32,
        connectome_kind="nws",
        train_ratio=0.7,
        val_to_test_ratio=0.5,
        split_plot_path=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(
            range(len(self.df)), train_size=train_ratio, random_state=42
        )
        self.val_index, self.test_index = train_test_split(
            self.val_index, train_size=val_to_test_ratio, random_state=42
        )
        if split_plot_path is not None:
            self._visualize_split(split_plot_path)
        # mean_birth_age_on_train_set used later for standardisation
        self.mean_scan_age_on_train_set = (
            self.df["scan_age"].iloc[self.train_index].mean()
        )
        self.n_ones = n_ones
        self.n_nodes = n_nodes
        self.node_pe = torch.ones((n_nodes, n_ones))

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
                    pd.read_csv(
                        os.path.join("connectomes-csv", file_name), header=None
                    ).to_numpy()
                )

        df = df.loc[has_connectome, :].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self, split_plot_path):
        sns.kdeplot(
            self.df["scan_age"].iloc[self.train_index], label="train", fill=True
        )
        sns.kdeplot(
            self.df["scan_age"].iloc[self.val_index], label="validation", fill=True
        )
        sns.kdeplot(self.df["scan_age"].iloc[self.test_index], label="test", fill=True)
        plt.title(
            f"Distribution of scan age across train (n={len(self.train_index)}), validation (n={len(self.val_index)})\n and test (n={len(self.test_index)}) sets"
        )
        plt.legend()
        plt.savefig(split_plot_path)
        plt.clf()

    def train_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.ones((self.n_nodes, self.n_ones)),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.train_index
            ],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.ones((self.n_nodes, self.n_ones)),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.val_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.ones((self.n_nodes, self.n_ones)),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.test_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )


class ScanAgeTrainValTestWithOneHot(LightningDataModule):
    def __init__(
        self,
        *,
        n_nodes=87,
        batch_size=32,
        connectome_kind="nws",
        train_ratio=0.7,
        val_to_test_ratio=0.5,
        split_plot_path=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(
            range(len(self.df)), train_size=train_ratio, random_state=42
        )
        self.val_index, self.test_index = train_test_split(
            self.val_index, train_size=val_to_test_ratio, random_state=42
        )
        if split_plot_path is not None:
            self._visualize_split(split_plot_path)
        # mean_birth_age_on_train_set used later for standardisation
        self.mean_scan_age_on_train_set = (
            self.df["scan_age"].iloc[self.train_index].mean()
        )
        self.n_nodes = n_nodes
        self.node_pe = torch.eye(n_nodes)

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
                    pd.read_csv(
                        os.path.join("connectomes-csv", file_name), header=None
                    ).to_numpy()
                )

        df = df.loc[has_connectome, :].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self, split_plot_path):
        sns.kdeplot(
            self.df["scan_age"].iloc[self.train_index], label="train", fill=True
        )
        sns.kdeplot(
            self.df["scan_age"].iloc[self.val_index], label="validation", fill=True
        )
        sns.kdeplot(self.df["scan_age"].iloc[self.test_index], label="test", fill=True)
        plt.title(
            f"Distribution of scan age across train (n={len(self.train_index)}), validation (n={len(self.val_index)})\n and test (n={len(self.test_index)}) sets"
        )
        plt.legend()
        plt.savefig(split_plot_path)
        plt.clf()

    def train_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.eye(self.n_nodes),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.train_index
            ],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.eye(self.n_nodes),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.val_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        torch.eye(self.n_nodes),
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    # subtracting the mean value causes better training perf.
                    # I do not do proper standardisation (div. by std.) to keep MSE and MAE values interpretable on their original scale
                    torch.tensor(
                        self.df["scan_age"].iloc[i] - self.mean_scan_age_on_train_set,
                        dtype=torch.float32,
                    ),
                )
                for i in self.test_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )


class SexTrainValTestWithLaplacian(LightningDataModule):
    def __init__(
        self,
        n_laplacian,
        *,
        n_nodes=87,
        batch_size=32,
        connectome_kind="nws",
        train_ratio=0.7,
        val_to_test_ratio=0.5,
        split_plot_path=None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_laplacian = n_laplacian

        self.df, self.connectomes = self._get_df_and_connectomes(connectome_kind)
        self.train_index, self.val_index = train_test_split(
            range(len(self.df)), train_size=train_ratio, random_state=42
        )
        self.val_index, self.test_index = train_test_split(
            self.val_index, train_size=val_to_test_ratio, random_state=42
        )
        if split_plot_path is not None:
            self._visualize_split(split_plot_path)

        # node_pe is (n_nodes, F) where F is the feature dimensionality
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
                    pd.read_csv(
                        os.path.join("connectomes-csv", file_name), header=None
                    ).to_numpy()
                )

        df = df.loc[has_connectome, :].reset_index(drop=True)
        connectomes = np.array(connectomes)
        return df, connectomes

    def _visualize_split(self, split_plot_path):
        sns.countplot(
            self.df["sex"].iloc[self.train_index],
            label="train",
            fill=True,
            stat="count",
            alpha=0.3,
        )
        print(self.df["sex"].iloc[self.train_index].value_counts())
        sns.countplot(
            self.df["sex"].iloc[self.val_index],
            label="val",
            fill=True,
            stat="count",
            alpha=0.3,
        )
        print(self.df["sex"].iloc[self.val_index].value_counts())
        sns.countplot(
            self.df["sex"].iloc[self.test_index],
            label="test",
            fill=True,
            stat="count",
            alpha=0.3,
        )
        print(self.df["sex"].iloc[self.test_index].value_counts())
        plt.title(
            f"Distribution of sex across train (n={len(self.train_index)}), validation (n={len(self.val_index)})\n and test (n={len(self.test_index)}) sets"
        )
        plt.legend()
        plt.savefig(split_plot_path)
        plt.clf()

    def _get_laplacian_pe(self, n_laplacian, n_nodes):
        with torch.no_grad():
            t = torch_geometric.transforms.AddLaplacianEigenvectorPE(k=n_laplacian)
            x = torch.eye(n_nodes, dtype=torch.float32)
            e = (
                torch.tensor(
                    [[i, j] for i in range(n_nodes) for j in range(n_nodes)],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )
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
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    torch.tensor(
                        self.df["sex"].map({"male": 0, "female": 1}).iloc[i],
                        dtype=torch.float32,
                    ),
                )
                for i in self.train_index
            ],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

    def val_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    torch.tensor(
                        self.df["sex"].map({"male": 0, "female": 1}).iloc[i],
                        dtype=torch.float32,
                    ),
                )
                for i in self.val_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        # due to some batching issues, I use the classic torch dataloader instead of the torch_geometric implementation
        return torch.utils.data.DataLoader(
            dataset=[
                (
                    (
                        self.node_pe.clone().detach(),  # a copy of the pos. encoding for each data point
                        torch.tensor(
                            self.connectomes[i, ...], dtype=torch.float32
                        ),  # adjacency matrix
                    ),
                    torch.tensor(
                        self.df["sex"].map({"male": 0, "female": 1}).iloc[i],
                        dtype=torch.float32,
                    ),
                )
                for i in self.test_index
            ],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )
