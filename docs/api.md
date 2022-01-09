# API
## labelme2token
```python
from dorec.apis import labelme2token

indir:<str> = "INPUT_DIR"
outdir:<str> = "OUTPUT_DIR"
labelme2token(
    indir, 
    outdir, 
    labels: <str, optional>,
    with_depth: <bool, optional>,
    noviz: <bool, optional>,
    num_keypoints: <int, optional>
    )
```

## augment
```python
from dorec.apis import augment

config_filepath = "CONFIG_FILE"
augment(config_filepath, out_dir: <str, optional>)
```