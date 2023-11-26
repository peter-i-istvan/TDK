from typing import Any
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import R2Score, PearsonCorrCoef, MeanAbsoluteError

class RegressionModel(LightningModule):
    def __init__(self, model, loss_fn, optim) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim

        self.val_correlation = PearsonCorrCoef()
        self.val_r2 = R2Score()
        self.val_mae = MeanAbsoluteError()

    def training_step(self, batch, _):
        g, y_true = batch
        x, adj = g
        y_hat = self.model(x, adj).squeeze(1)

        loss = self.loss_fn(y_hat, y_true)
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch):
        g, y_true = batch
        x, adj = g
        y_hat = self.model(x, adj).squeeze(1)

        loss = self.loss_fn(y_hat, y_true)
        self.log("val_loss", loss)
        
        self.val_correlation.update(y_hat, y_true)
        self.val_r2.update(y_hat, y_true)
        self.val_mae.update(y_hat, y_true)
        
        return loss
    
    def on_validation_epoch_end(self):
        corr = self.val_correlation.compute()
        r2 = self.val_r2.compute()
        mae = self.val_mae.compute()

        self.log_dict({"val_correlation": corr, "val_r2": r2, "val_mae": mae})
    
    def configure_optimizers(self):
        return self.optim
