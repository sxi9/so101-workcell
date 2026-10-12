# SO-101 Workcell

Language-driven pick-and-place workcell for the SO-101 arm. Built in MuJoCo on a laptop
(RTX 3050 6 GB), with the same control code targeting a Cyberwave digital twin.

> Work in progress. Sections fill in as each layer lands.

## Layers

| Layer | Folder | Status |
|---|---|---|
| Simulation: SO-101 MJCF, tabletop scene, IK, scripted skills | `so101_workcell/sim` | scaffolded |
| RL: grasp-and-lift env with domain randomization, SAC | `so101_workcell/rl` | planned |
| Agent: VLM decides *what*, skills decide *how* | `so101_workcell/agent` | planned |
| Hub: LeRobot dataset export, push models/datasets to Hugging Face | `so101_workcell/hub` | planned |
| Cyberwave backend: same skills on the Playground twin | `so101_workcell/sim/backends` | planned |

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

## Credits

SO-101 model from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
