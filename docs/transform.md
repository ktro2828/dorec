# Transforms
now we support data transformation shown below,
- Compose
- RandomApply
- ToTensor
- Resize
- ColorJitter
- EdgeFilter
- RandomPerspective
- RandomCrop
- RandomRotate
- RandomFlip
- VerticalFlip
- HorizontalFlip
- RandomNoise
- RandomErase

## How to use
you can build registerd transforms with `build_transforms()`.

**to apply transforms, call it in dataset object**

```python
from dorec.core import Config
from dorec.datasets import build_transforms

cfg = Config("CONFIG_FILE")
train_dataset_cfg = cfg.train.copy()
test_dataset_cfg = cfg.test.copy()
# [OPTIONAL] val_dataset_cfg = cfg.val.copy()

train_pipelines = train_dataset_cfg.pipelines
# If compose=True, returns composed transforms
# else, returns listed transform objects
train_transforms = build_transforms(
    pipelines=train_pipelines,
    compose=<bool, optional>
    )
```

## Customization
```python
from dorec.core import TRANSFORMS
from dorec.datasets import TransformBase

@TRANSFORMS.register()
class MyTransform(TransformBase):
    def __init__(self, name=None):
        super(TransformBase, self).__init__(name)

    def __call__(self, data):
        """
        Args:
            data [dict[str, any]]:
                - inputs <np.ndarray>
                - targets <dict[str, np.ndarray]
        """
        # === DO SOMETHING ===

        return data
```