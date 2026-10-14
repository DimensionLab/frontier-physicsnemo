# frontier-physicsnemo

Runs NVIDIA PhysicsNeMo 2.x models in Frontier. You train, predict, evaluate and fine-tune
them with the same commands as Frontier's own models, and the weights stay loadable in
PhysicsNeMo.

## Install

Install the wheel from the release bundle:

```bash
pip install --no-index --find-links wheels frontier-physicsnemo
```

The models need `nvidia-physicsnemo>=2.2,<3` and `torch_geometric` (the `physicsnemo`
extra). Without them the models are listed, and building one fails with an install hint.

## Example

Set `model.name` in a training config:

```yaml
model:
  name: physicsnemo.transolver
  params: { n_layers: 8, n_hidden: 256, slice_num: 64 }
```

```bash
python -m frontier_train train --config my-config.yaml
python -m frontier_train finetune --base nvidia-transolver-drivaerml --config my-config.yaml
```

## Models

| `model.name`                | PhysicsNeMo class                    | Task                    |
| --------------------------- | ------------------------------------ | ----------------------- |
| `physicsnemo.transolver`    | `models.transolver.Transolver`       | Surface pressure and Cd |
| `physicsnemo.geotransolver` | `models.geotransolver.GeoTransolver` | Surface pressure and Cd |
| `physicsnemo.figconvnet`    | `models.figconvnet.FIGConvUNet`      | Surface pressure and Cd |
| `physicsnemo.meshgraphnet`  | `models.meshgraphnet.MeshGraphNet`   | Transient mesh rollouts |

`model.params` takes PhysicsNeMo's constructor arguments. Input and output widths come from
the dataset.

## API

| Module                             | What it does                                                                                                                                                |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frontier_physicsnemo.checkpoints` | `save_mdlus`, `load_mdlus`, `read_mdlus`: write and read `.mdlus` checkpoints. A training run of a PhysicsNeMo model writes `best.mdlus` next to `best.pt`. |
| `frontier_physicsnemo.hydra`       | Convert PhysicsNeMo recipe model nodes (`_target_` plus arguments) to Frontier `model:` blocks and back.                                                    |
| `frontier_physicsnemo.compat`      | `ensure_torch_scatter()`: provide `scatter` and `segment_csr` when `torch_scatter` is not installed.                                                        |
| `frontier_physicsnemo.logging`     | `FrontierLogger`: log a PhysicsNeMo training loop to Frontier.                                                                                              |

## Licence

PhysicsNeMo and NVIDIA's pretrained weights keep their own licences. Read the licence of a
pretrained model with `frontier hub info <model>` before you use it.
