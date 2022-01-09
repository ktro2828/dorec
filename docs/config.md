# Config file

```
# Task
task <str, sequece[str]>: ...task name(s)

# Running paramers
parameters:
    work_dir <str, optional>: ...working directory path, results will be saved(checkpoints and visualize results e.t.c) (default: ./experiments)
    checkpoint <str, optional>: ...checkpoint path (default: None)
    train:
        max_epoch <int>: ...maximum epoch to be trained
        device <str, optional>: ...device which will be tensor on(default: cpu)
        gpu_ids <str, optional>: gpu ids (default: 0)
        batch_size <int, optional>: batch size (default: 1)
    test:
        device <str, optional>
        gpu_ids <str, optional>
        batch_size <int, optional>

# Model config
model:
    # In case of building model with specifing ``name``
    name <str>: ...model name
    in_channels <int>: ...the number of input channels
    num_classes <int>: ...the number of output channels
    # In case of building model with ``backbones`` and ``heads``
    backbones:
        name <str>: ...backbone name
        in_channels <int>: ...the number of input channels
        pretrain <bool, optional>: ...if use pretrained backbone (default: True)
    heads:
        name <str>: ...head name
        fc_dim <int>: ...the number of feature map dimensions
        deep_sup <bool, optional>: ...if running with deep suppresion (default: False)

# Dataset config
dataset:
    name <str>: ...dataset object name
    input_type <str>: ...input type name
    use_dims <int>: ...the number of using dimensions as input
    **kwargs
    train:
        root <str, sequene[str]>: ...path(s) of data
        pipelines <list[dict]>:
            - name <str>: ...transform name
              **kwargs
            ...
    test:
        root <str, sequene[str]>
        pipelines <list[dict]>
    val[optinoal]: ...if not specified, train config will be loaded

# Loss config
loss:
    <TASK_NAME>:
        name <str>: ...loss name
        **kwargs

# Evaluation config
evaluation:
    <TASK_NAME>:
        methods <str, sequence[str], optional>: evaluation method name(s)
        **kwargs

# Optimizer config
optimizer:
    name <str>: ...optimizer name
    **kwargs

# Scheduler config
scheduler:
    name <str>: ...lr_scheduler name
    **kwargs
```