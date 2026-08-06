# HTCC
Hierarchical Team Collaborative Cognition for Cooperative MARL

HTCC extends the Multi-Agent Transformer (MAT) with an explicit team
collaborative cognition module. It aggregates agent-wise interaction
representations into team context, builds hierarchical team cognition, and
feeds the shared cognition back to individual agents through gated feedback.

## Installation

### Dependences
``` Bash
pip install -r requirements.txt
```

### Multi-agent MuJoCo
Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:
``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### StarCraft II & SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.

### Google Research Football
Please following the instructios in https://github.com/google-research/football. 

### Bi-DexHands 
Please following the instructios in https://github.com/PKU-MARL/DexterousHands. 

## How to run
After the required environments are installed, run the scripts in
`memat/scripts`. For example, to run HTCC on one SMAC map:
```bash
sh memat/scripts/train_smac_variant.sh \
  --map 5m_vs_6m --algo nest_mat_clock --seed 1 --gpu 0
```

For `nest_mat_clock`, omitting `--clock-beta` automatically selects the
map-specific value reported in Table 3 of the paper:

| SMAC map | beta |
| --- | ---: |
| `1c3s5z` | 0.25 |
| `3s5z` | 0.50 |
| `5m_vs_6m` | 0.75 |
| `8m_vs_9m` | 0.00 |
| `10m_vs_11m` | 0.25 |
| `6h_vs_8z` | 0.50 |
| `3s5z_vs_3s6z` | 1.00 |
| `MMM2` | 0.50 |
| `27m_vs_30m` | 0.50 |

The default can be overridden explicitly, for example with
`--clock-beta 0.25`. Multiple maps and seeds can be launched with
`run_smac_variants.sh`.

Other benchmark scripts are provided for Google Research Football,
Multi-Agent MuJoCo, and Bi-DexHands. Training outputs are written under the
script directory and are intentionally excluded from this repository copy.

