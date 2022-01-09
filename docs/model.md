# Model
now we're supporting models shown below

- Models
  - UNet
  - SegNet
- BackBones
  - HRNetV2
  - MobileNetV2
  - ResNet18/34/50/101
  - ResNext50/101
- Heads
  - CNHead
  - DualHead
  - MultiHead
  - PPM
  - UPerNet

## How to use
you can build registered models with `build_model()`

```python
from dorec.core import Config
from dorec.models import build_model

cfg = Config("CONFIG_FILE")

# If assign_task=True, model outputs dict[str, torch.Tensor] mapped by TASK NAMES
# else, outputs torch.Tensor or tuple[torch.Tensor]
# ex.1: build with original config
model = build_model(
    cfg=cfg
    task=<tuple[str], optional>
    assign_task=<bool, optional>
)
# ex.2: build with config.model
model_cfg = cfg.model.copy()
task = cfg.task
model = build_model(
    cfg=model_cfg,
    task=task,  # assign_task=False, unused
    assign_task=<bool, optional>
)
```

optionally, you can only build registered `backbones` and `heads` with `build_backbone()` and `build_head()`.

```python
from dorec.core import Config
from dorec.models import build_backbone, build_head

cfg = Config("CONFIG_FILE")

# Build backbone
backbone_cfg = cfg.model.backbones.copy()
backbone = build_backbone(backbone_cfg)

# Build head
head_cfg = cfg.model.heads.copy()
head = build_head(head_cfg)

# Wrap backbone and head by EDModule(;Encoder Decoder Module)
from dorec.models import EDModule
model = EDModule(backbone, head)

# Assin task to model wrapping by TaskModule
from dorec.models import TaskModule

task = cfg.task
model = TaskModule(task,model)
```
## Customization

### Models
the models registered with `MODELS` will be built only specifying `name` in config file

```python
from dorec.core import MODELS
from dorec.models import ModuleBase

@MODELS.register()
class MyModel(ModuleBase):
    def __init__(self, in_channels, num_classes):
        super(MyModel, self).__init__()
        # === DEFINE INSTANCES ===

    def forward(self, x):
        """
        Args:
            x (torch.Tensor)
        Returns:
            out (torch.Tensor, tuple[torch.Tensor])
        """
        # === DO SOMETHING ===
        out: torch.Tensor, tuple[torch.Tensor]
        return out
```

### Backbones & Heads
the models registered with `BACKBONES` and `HEADS` will be built specifying like below in config file
```
model:
    backbones:
        name: ...
    heads:
        name: ...
```

#### **Backbones**
```python
from dorec.core import BACKBONES
from dorec.models import ModuleBase

@BACKBONES.register()
class MyBackbone(ModuleBase):
    def __init__(self, in_channels, **kwargs):
        super(MyBackbone, self).__init__()
        # === DEFINE INSTANCES ===

    def forward(self, x):
        """
        Args:
            x (torch.Tensor)
        Returns:
            out (list[torch.Tensor])
        """
        # === DO SOMETHIG ===
        out: list[torch.Tensor]
        return out
```

#### **Heads**
```python
from dorec.core import HEADS
from dorec.models import HeadBase

@HEADS.register()
class MyHead(HeadBase):
    def __init__(self, deep_sup=False, **kwargs):
        super(MyHead, self).__init__(deep_sup)
        # === DEFINE INSTANCES ===

    def forward(self, features, segSize):
        """
        Args:
            features (list[torch.Tensor])
            segSize (tuple[int])
        Returns:
            out (torch.Tensor, tuple[torch.Tensor])
        """
        # === DO SOMETHIG ===
        # Resize out with F.interpolate(out, segSize)
        out: torch.Tensor, tuple[torch.Tensor]
        if deep_sup:
            out: list[torch.Tensor]

        return out
```

