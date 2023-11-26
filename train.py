import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt

# https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
seed_everything(42, workers=True)

from data import BirthAgeWithLaplacian
from models import RegressionModel, BaselineGCN


def birth_age_regression():    
    gcn = BaselineGCN(input_channels=10, hidden_conv_channels=20, out_conv_channels=5, hidden_mlp_features=10)
    print(gcn)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model = RegressionModel(gcn, loss_fn, optim)
    datamodule = BirthAgeWithLaplacian(10, batch_size=32)
    wandb_logger = WandbLogger(project="Onlab2-BA")
    trainer = Trainer(
        max_epochs=300, 
        log_every_n_steps=50, 
        logger=wandb_logger, 
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)]
    )

    trainer.fit(model, datamodule=datamodule)

    # Plot predictions on validation set
    with torch.no_grad():
        n_val = len(datamodule.val_index)
        x = datamodule.node_pe.clone().detach().unsqueeze(0).repeat(n_val, 1, 1)
        adj = torch.tensor(datamodule.connectomes[datamodule.val_index,...], dtype=torch.float32)
        y_hat = gcn(x, adj).squeeze().numpy() + datamodule.mean_birth_age_on_train_set
        y_true = datamodule.df["birth_age"].iloc[datamodule.val_index].to_numpy()
        min_, max_ = min(y_hat.min(), y_true.min()), max(y_hat.max(), y_true.max())

        plt.title("Prediction of birth age on the validation set")
        plt.scatter(y=y_hat, x=y_true)
        plt.plot([min_, max_], [min_, max_], c="r", ls="--", label="ideal prediction")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.legend()
        plt.savefig("prediction.png")
        plt.clf()

    
def sex_classification():
    ...

if __name__ == "__main__":
    birth_age_regression()
    # sex_classification()
