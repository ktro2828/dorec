# Experiment
## Train
- from API
```python
from dorec.core import Config
from dorec.apis import train

cfg = Config("CONFIG_FILE")
train(cfg)
```
- from source
```
# [] is optional
$ python tools/train.py --config <CONFIG_FILE> [--checkpoint <CHECKPOINT_FILE>]
```

### **Optimizer**
we support optimizer shown below, they are implemented in `torch.optim` originaly.
- Adam
- SGD
- RMSprop
- Adadelta
- AdamW

### **lr_scheduler**
we support lr_scheduler shown below, they are implemented in `torch.optim.lr_scheduler`
- MultiStepLR
- ExponentialLR
- ReduceLROnPlateau
- StepLR
- CosineAnnealingLR
- CyclicLR
- LambdaLR

## Test
- from API
```python
from dorec.core import Config
from dorec.apis import test

cfg = Config("CONFIG_FILE")
test(cfg)
```
- from source
```
$ python tools/test.py --config <CONFIG_FILE> --checkpoint <CHEKPOINT_FILE>
```

## Inference
- from API
```python
from dorec.core import Config
from dorec.apis import inference

cfg = Config("CONFIG_FILE")
inferece(cfg)
```
- from source
```
# [] is optional, **at least one of the ``--rgb`` or ```--depth`` must be specified**
$ python tools/inference.py --config <CONFIG_FILE> --checkpoint <CHECKPOINT_FILE> [--rgb <RGB_DIR> --depth <DEPTH_DIR>]
```

## Show data
- from API
```python
from dorec.core import Config
from dorec.apis import show_data

cfg = Config("CONFIG_FILE")
show_data(cfg, max_try=<int>, is_test=<bool>)
```

- from source
```
# [] is optional
$ python tools/show_data.py --config <CONFIG_FILE> -n <NUM_TRY> [--is_test]
```