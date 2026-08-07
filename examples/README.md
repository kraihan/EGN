# Examples

Smallest first. Each script is self-contained and generates its own data, so they run anywhere.

| Script | Shows |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | five ways to use EGN, in increasing order of control |
| [`02_bring_your_own_data.py`](02_bring_your_own_data.py) | every input convention, one model class |
| [`03_custom_architecture.py`](03_custom_architecture.py) | building a trunk by hand, and training it |
| [`04_train_ddp.py`](04_train_ddp.py) | multi-GPU with `torchrun` |

```bash
python examples/01_quickstart.py
torchrun --nproc_per_node=4 examples/04_train_ddp.py --epochs 50
```
