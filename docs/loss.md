# Loss
now we're supporting losses shown below

## How to use
you can build registered losses with `build_loss()`
```python
from dorec.core import Config
from dorec.losses import build_loss

cfg = Config("CONFIG_FILE")
loss_cfg = cfg.loss.copy()

loss = build_loss(loss_cfg)
```

## Customization
```python
import torch.nn as nn

from dorec.core import LOSSES

@LOSSES.register()
class MyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(MyLoss, self).__init__()
        # === DEFINE INSTANCES ===

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor)
            targets (torch.Tensor)
        Returns:
            loss (torch.Tensor)
        """
        # === DO SOMETHIN ===
        loss: torch.Tensor
        return loss
```