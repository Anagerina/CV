import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from segmentation_models_pytorch.losses import FocalLoss
from config import NUM_CLASSES
from config import WEIGHTS

weights_tensor = torch.tensor(WEIGHTS, dtype=torch.float32)


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.focal = FocalLoss(mode='multiclass', reduction='none')
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.focal(logits, targets)
        weights = self.class_weights.to(targets.device)
        batch_weights = weights[targets.long()]
        return (loss * batch_weights).mean()



class UNetModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.criterion = WeightedFocalLoss(class_weights=WEIGHTS)  
        self.model = model
        self.learning_rate = learning_rate
        self.n_classes = NUM_CLASSES
        self.iou_history = {cls: [] for cls in range(self.n_classes)}
        self._epoch_batch_ious = []


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self.forward(images)
        loss = self.criterion(logits, masks)
        
        preds = torch.argmax(logits, dim=1)
        iou, _ = self.compute_iou(preds, masks)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        logits = self.forward(images)
        
        loss = self.criterion(logits, masks)
        
        preds = torch.argmax(logits, dim=1)
        iou, class_ious_list = self.compute_iou(preds, masks)
        
        self._epoch_batch_ious.append(class_ious_list)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        if not self._epoch_batch_ious:
            return
            
        for cls in range(self.n_classes):
            cls_values = [batch_list[cls] for batch_list in self._epoch_batch_ious]
            mean_epoch_iou = sum(cls_values) / len(cls_values)
            self.iou_history[cls].append(round(mean_epoch_iou, 4))
            
        self._epoch_batch_ious.clear()



    
    def test_step(self, batch, batch_idx):
        images, masks = batch
        
        logits = self.forward(images)
        
        loss = self.criterion(logits, masks)
        
        preds = torch.argmax(logits, dim=1)
        iou, _ = self.compute_iou(preds, masks)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def compute_iou(self, preds, targets):
        ious = []
        for cls in range(self.n_classes):
            pred_cls = (preds == cls)
            target_cls = (targets == cls)
            intersection = (pred_cls & target_cls).float().sum((1, 2))
            union = (pred_cls | target_cls).float().sum((1, 2))
            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou.mean().item())
        return torch.tensor(ious).mean(), ious
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = self.trainer.max_epochs * 1000
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate * 10,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }












