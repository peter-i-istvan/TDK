import torch
from lightning import Trainer, seed_everything
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
seed_everything(42, workers=True)

from data import (
    BirthAgeWithLaplacian,
    BirthAgeTrainValTestWithLaplacian,
    ScanAgeTrainValTestWithLaplacian,
    ScanAgeTrainValTestWithLocalDegreeProfile,
    SexTrainValTestWithLaplacian,
    ScanAgeTrainValTestWithOnes,
    ScanAgeTrainValTestWithOneHot,
    BirthAgeTrainValTestWithOneHot,
)
from models import RegressionModel, ClassificationModel, BaselineGCN


def plot_final_predictions_1(gcn, datamodule, save_to):
    with torch.no_grad():
        n_val = len(datamodule.val_index)
        x = datamodule.node_pe.clone().detach().unsqueeze(0).repeat(n_val, 1, 1)
        adj = torch.tensor(
            datamodule.connectomes[datamodule.val_index, ...], dtype=torch.float32
        )
        y_hat = gcn(x, adj).squeeze().numpy() + datamodule.mean_birth_age_on_train_set
        y_true = datamodule.df["birth_age"].iloc[datamodule.val_index].to_numpy()
        min_, max_ = min(y_hat.min(), y_true.min()), max(y_hat.max(), y_true.max())

        plt.title(f"Prediction of birth age on the validation (n={n_val}) set")
        plt.scatter(y=y_hat, x=y_true)
        plt.plot([min_, max_], [min_, max_], c="r", ls="--", label="ideal prediction")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.legend()
        plt.savefig(save_to)
        plt.clf()


def plot_final_predictions_2(
    gcn,
    datamodule,
    target_variable,
    save_to,
    pos_enc,
):
    with torch.no_grad():
        eval_index = np.concatenate([datamodule.val_index, datamodule.test_index])
        n_eval = len(eval_index)

        x = datamodule.node_pe.clone().detach().unsqueeze(0).repeat(n_eval, 1, 1)
        adj = torch.tensor(datamodule.connectomes[eval_index, ...], dtype=torch.float32)
        y_hat = gcn(x, adj).squeeze().numpy() + datamodule.mean_birth_age_on_train_set
        y_true = datamodule.df[target_variable].iloc[eval_index].to_numpy()
        min_, max_ = min(y_hat.min(), y_true.min()), max(y_hat.max(), y_true.max())

        plt.title(
            f"{pos_enc}\nPrediction of {target_variable} on the validation and test (n={n_eval}) set"
        )
        plt.scatter(y=y_hat, x=y_true)
        plt.plot([min_, max_], [min_, max_], c="r", ls="--", label="ideal prediction")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.legend()
        plt.savefig(save_to)
        plt.clf()


def birth_age_regression():
    """Regression with okay-ish model, but using early stopping and only working on train-val split (no test)"""
    gcn = BaselineGCN(
        input_channels=10,
        hidden_conv_channels=20,
        out_conv_channels=5,
        hidden_mlp_features=10,
    )
    # print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = BirthAgeWithLaplacian(
        10, batch_size=32, split_plot_path="plots/birth_age_train_val_split.png"
    )
    wandb_logger = WandbLogger(project="Onlab2-BA")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()

    # # Plot predictions on validation set
    plot_final_predictions_1(
        gcn, datamodule, save_to="plots/birth_age_predictions_val.png"
    )


def birth_age_regression_train_val_test_laplacian():
    gcn = BaselineGCN(
        input_channels=10,
        hidden_conv_channels=20,
        out_conv_channels=5,
        hidden_mlp_features=10,
    )
    print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = BirthAgeTrainValTestWithLaplacian(
        10, batch_size=32, split_plot_path="plots/birth_age_train_val_test_split.png"
    )
    wandb_logger = WandbLogger(project="Onlab2-BA")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    # Plot predictions on validation set
    plot_final_predictions_2(
        gcn,
        datamodule,
        "birth_age",
        f"plots/birth_age_predictions_val_test.png",
        pos_enc="LAPLACIAN10",
    )


def birth_age_regression_train_val_test_one_hot():
    gcn = BaselineGCN(
        input_channels=87,
        hidden_conv_channels=20,
        out_conv_channels=5,
        hidden_mlp_features=10,
    )
    print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = BirthAgeTrainValTestWithOneHot(
        batch_size=32, split_plot_path="plots/birth_age_train_val_test_split.png"
    )
    wandb_logger = WandbLogger(project="Onlab2-BA")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    # Plot predictions on validation set
    plot_final_predictions_2(
        gcn,
        datamodule,
        "birth_age",
        f"plots/birth_age_predictions_one_hot_val_test.png",
        pos_enc="ONE_HOT",
    )


def scan_age_regression_train_val_test_laplacian():
    gcn = BaselineGCN(
        input_channels=10,
        hidden_conv_channels=20,
        out_conv_channels=5,
        hidden_mlp_features=10,
    )
    # print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = ScanAgeTrainValTestWithLaplacian(
        10,
        batch_size=32,
        split_plot_path="plots/scan_age_train_val_test_split.png",
    )
    wandb_logger = WandbLogger(project="Onlab2-SA")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    # Plot predictions on validation set
    plot_final_predictions_2(
        gcn,
        datamodule,
        "scan_age",
        save_to="plots/scan_age_predictions_laplacian_val_test.png",
        pos_enc="LAPLACIAN10",
    )


def scan_age_regression_train_val_test_local_degree_profile():
    gcn = BaselineGCN(
        input_channels=5,
        hidden_conv_channels=20,
        out_conv_channels=5,
        hidden_mlp_features=10,
    )
    # print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = ScanAgeTrainValTestWithLocalDegreeProfile(
        batch_size=32,
        split_plot_path="plots/scan_age_train_val_test_split.png",
    )
    wandb_logger = WandbLogger(project="Onlab2-SA")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    # Plot predictions on validation set
    plot_final_predictions_2(
        gcn,
        datamodule,
        "scan_age",
        save_to="plots/scan_age_predictions_ldp_val_test.png",
        pos_enc="LOCAL_DEGREE_PROFILE",
    )


def scan_age_regression_train_val_test_ones_10():
    gcn = BaselineGCN(
        input_channels=10,
        hidden_conv_channels=20,
        out_conv_channels=5,
        hidden_mlp_features=10,
    )
    # print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = ScanAgeTrainValTestWithOnes(
        10,
        batch_size=32,
        split_plot_path="plots/scan_age_train_val_test_split.png",
    )
    wandb_logger = WandbLogger(project="Onlab2-SA")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    # Plot predictions on validation set
    plot_final_predictions_2(
        gcn,
        datamodule,
        "scan_age",
        save_to="plots/scan_age_predictions_ones_10_val_test.png",
        pos_enc="ONES10",
    )


def scan_age_regression_train_val_test_ones_87():
    gcn = BaselineGCN(
        input_channels=87,
        hidden_conv_channels=20,
        out_conv_channels=5,
        hidden_mlp_features=10,
    )
    # print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = ScanAgeTrainValTestWithOnes(
        87,
        batch_size=32,
        split_plot_path="plots/scan_age_train_val_test_split.png",
    )
    wandb_logger = WandbLogger(project="Onlab2-SA")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    # Plot predictions on validation set
    plot_final_predictions_2(
        gcn,
        datamodule,
        "scan_age",
        save_to="plots/scan_age_predictions_ones_87_val_test.png",
        pos_enc="ONES87",
    )


def scan_age_regression_train_val_test_one_hot():
    gcn = BaselineGCN(
        input_channels=87,
        hidden_conv_channels=20,
        out_conv_channels=5,
        hidden_mlp_features=10,
    )
    # print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = ScanAgeTrainValTestWithOneHot(
        batch_size=32,
        split_plot_path="plots/scan_age_train_val_test_split.png",
    )
    wandb_logger = WandbLogger(project="Onlab2-SA")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    # Plot predictions on validation set
    plot_final_predictions_2(
        gcn,
        datamodule,
        "scan_age",
        save_to="plots/scan_age_predictions_one_hot_val_test.png",
        pos_enc="ONE_HOT",
    )


def sex_classification():
    gcn = BaselineGCN(
        input_channels=5,
        hidden_conv_channels=5,
        out_conv_channels=2,
        hidden_mlp_features=5,
        final_sigmoid=True,
    )
    print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    model = ClassificationModel(gcn, loss_fn, optim)
    datamodule = SexTrainValTestWithLaplacian(
        5, batch_size=32, split_plot_path="plots/sex_train_val_test_split.png"
    )
    wandb_logger = WandbLogger(project="Onlab2-S")
    trainer = Trainer(
        max_epochs=800,
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100)],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    # Plot predictions on validation set
    # plot_final_predictions_3(gcn, datamodule, "scan_age")


if __name__ == "__main__":
    # Run these two one after another to reproduce the good results.
    # birth_age_regression()
    # birth_age_regression_train_val_test_laplacian()
    # birth_age_regression_train_val_test_one_hot()
    # The following runs should be started on their own
    # idea: make baseline model bigger and bigger until it reaches just past the train dimensionality
    # track overfitting tendencies, and give estimate to the optimal model size
    # Scan age - run this on its own to reproduce the results
    # scan_age_regression_train_val_test_laplacian()
    # scan_age_regression_train_val_test_local_degree_profile()
    # scan_age_regression_train_val_test_ones_10()
    # scan_age_regression_train_val_test_ones_87()
    scan_age_regression_train_val_test_one_hot()
    # sex_classification()
