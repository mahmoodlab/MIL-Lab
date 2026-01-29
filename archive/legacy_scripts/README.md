# Legacy Scripts

These scripts contain hardcoded paths and were archived during the refactoring to use config-driven training.

## Why Archived

These scripts have hardcoded paths to specific data directories (e.g., `/media/nadim/Data/...`) and are not portable. They were used during early development and experimentation.

## Replacement Scripts

Use these config-driven scripts instead:

| Legacy Script | Replacement |
|--------------|-------------|
| `run_mil.py` | `train_mil.py --config config.json` |
| `train_abmil_simple.py` | `train_mil.py --model abmil.base.uni_v2.pc108-24k` |
| `train_clam_simple.py` | `train_mil.py --model clam.base.uni_v2.none` |
| `run_mil_experiments_gem3.py` | `run_mil_experiments.py --config config.json` |
| `train.sh`, `run_cv.sh` | `run_mil_experiments_cv.py --config config.json` |

## If You Need These

If you need to use these scripts, update the hardcoded paths at the top of each file to match your data locations.
