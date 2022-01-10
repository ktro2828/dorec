# dorec
dorec is usefull pipelines for Deformable Object Recognition tasks

# Support tasks
- [x] Sementic segmentation
- [x] Edge detection
- [x] Depth estimation
- [x] Keypoint detection

# Dependencies
- Pytorch >= 1.4

# Install
```
$ pip install -e .
```

# Experiments
- `dorec` is supporting running experiments (train/test/inference...) from config file

    more detail, see [docs/config.md](docs/config.md)

- for train/test/inference/show_data, you can run from API or tools/XXX.py

    more detail, see [docs/experiment.md](docs/experiment.md)

# Customize
- you can customize `dataset`, `transform`, `model`, `loss`

    more detail, see 
    - [docs/dataset.md](docs/dataset.md)
    - [docs/transform.md](docs/transform.md)
    - [docs/model.md](docs/model.md)
    - [docs/loss.md](docs/loss.md)