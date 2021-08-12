import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        mixup_fn=None,
        lr=1e-3,
        weight_decay=1e-5,
        scheduler="cosine",
        cosine_max_iters=2000,
        plateau_factor=0.2,
        plateau_patience=2,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "mixup_fn"])
        self.model = model
        self.mixup_fn = mixup_fn

        # if mixup_alpha > 0.:
        #     self.mixup_fn = Mixup(mixup_alpha, prob=0.5, num_classes=model.num_classes)
        #     self.train_loss_fn = SoftTargetCrossEntropy()
        # else:
        #     self.train_loss_fn = nn.CrossEntropyLoss()

        self.train_loss_fn = SoftTargetCrossEntropy() if mixup_fn else nn.CrossEntropyLoss()

        # self.train_acc_top1 = Accuracy()
        self.val_acc_top1 = Accuracy()
        self.test_acc_top1 = Accuracy()

        # self.train_acc_top5 = Accuracy(top_k=5)
        self.val_acc_top5 = Accuracy(top_k=5)
        self.test_acc_top5 = Accuracy(top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mixup_fn:
            x, y = self.mixup_fn(x, y)
        y_pred = self(x)
        loss = self.train_loss_fn(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # self.log('train_acc_top1', self.train_acc_top1(y_pred, y), on_step=True, on_epoch=False, prog_bar=True)
        # self.log('train_acc_top5', self.train_acc_top5(y_pred, y), on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        top1_acc = self.val_acc_top1(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc_top1", top1_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc_top5", self.val_acc_top5(y_pred, y), on_step=False, on_epoch=True, prog_bar=True)
        self.log("hp_metric", top1_acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc_top1", self.test_acc_top1(y_pred, y), on_step=False, on_epoch=True)
        self.log("test_acc_top5", self.test_acc_top5(y_pred, y), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.hparams.cosine_max_iters, eta_min=self.hparams.lr * 1e-2)
            scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        elif self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, factor=self.hparams.plateau_factor, patience=self.hparams.plateau_patience
            )
            scheduler_dict = {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
