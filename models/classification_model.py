from lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall


class ClassificationModel(LightningModule):
    def __init__(self, model, loss_fn, optim) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim

        self.val_acc = Accuracy("binary")
        self.val_prec = Precision("binary")
        self.val_rec = Recall("binary")

        self.test_acc = Accuracy("binary")
        self.test_prec = Precision("binary")
        self.test_rec = Recall("binary")

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

        self.val_acc.update(y_hat, y_true)
        self.val_prec.update(y_hat, y_true)
        self.val_rec.update(y_hat, y_true)

        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        prec = self.val_prec.compute()
        rec = self.val_rec.compute()

        self.log_dict({"val_acc": acc, "val_prec": prec, "val_rec": rec})

    def test_step(self, batch):
        g, y_true = batch
        x, adj = g
        y_hat = self.model(x, adj).squeeze(1)

        loss = self.loss_fn(y_hat, y_true)
        self.log("test_loss", loss)

        self.test_acc.update(y_hat, y_true)
        self.test_prec.update(y_hat, y_true)
        self.test_rec.update(y_hat, y_true)

        return loss

    def on_test_epoch_end(self):
        acc = self.test_acc.compute()
        prec = self.test_prec.compute()
        rec = self.test_rec.compute()

        self.log_dict({"test_acc": acc, "test_prec": prec, "test_rec": rec})

    def configure_optimizers(self):
        return self.optim
