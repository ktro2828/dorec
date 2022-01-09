# Dataset
we're assuming build dataset generated with [ktro2828/Turneup-Scene-Generator](https://github.com/ktro2828/Turnedup-Scene-Generator.git)

now, we support `ImageTaskDataset` and `KeypointTaskDataset`
- `ImageTaskDataset`:

    dataet for tasks including `semantic segmentation`, `edge estimation` and `depth estimation`
    
- `KeypointTaskDataset`

    dataset for tasks including `keypoint detection`

## How to use
you can build registered datasets with `build_dataset()`

```python
from dorec.core import Config
from dorec.datasets import build_dataset

cfg = Config("CONFIG_FILE")
train_dataset_cfg = cfg.dataset.train.copy()
test_dataset_cfg = cfg.dataset.test.copy()
# [OPTIONAL] val_dataset_cfg = cfg.dataset.val.copy()

train_dataset_cfg.input_type = cfg.dataset.input_type
train_dataset_cfg.use_dims = cfg.dataset.use_dims

train_dataset = build_dataset(
    train_dataset_cfg, 
    input_type=<str, optional>
    use_dims=<int, optional>
    )
```

## Customization
```python
from dorec.core import DATASETS
from dorec.datasets import CustomDatasetBase

@DATASETS.register()
class MyDataset(CustomDatasetBase):
    def __init__(self, task, root, input_type, use_dims, pipelines):
        super(MyDataset, self).__init__(task, root, input_type, use_dims, pipelines)

    def __getitem__(self, idx):
        # === DO SOMETHIG ====

        inputs <torch.Tensor>: input tensor
        targets <dict[str, torch.Tensor]>: tensor of targets, which mapped by "TASK NAME"

        data = {"inputs": inputs, "targets": targets}

        return data

    def __len__(self):
        # === RETURN DATASET LENGTH ===
```