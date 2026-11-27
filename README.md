# Hardware_prune_-and-Cyber-Physical-Co-Design-for-Real-Time
Pruning means selectively cutting away unwanted parts—such as branches from a plant—or reducing excess items to improve health, shape, or efficiency
# Hardware Pruning and Cyber-Physical Co-Design for Real-Time Systems
 
> Pruning means selectively cutting away unwanted parts, such as branches from a plant, or reducing excess items to improve health, shape, or efficiency. Here, the idea is applied to neural networks: remove redundant weights and channels so models run faster and leaner on real-time, resource-constrained hardware.
 
---
 
## Overview
 
This repository explores **structured pruning of a YOLO classification model** (`yolo26n-cls.pt`) and how pruning decisions can be co-designed with the target cyber-physical platform (latency budget, memory, power) for real-time inference.
 
The goal is to shrink the model while keeping accuracy acceptable, so it can meet hard real-time deadlines on embedded or edge hardware.
 
## Key Ideas
 
- **Model pruning**: remove unneeded channels/layers from a pretrained YOLO classifier.
- **Hardware-aware design**: choose what to prune based on real constraints (latency, memory, compute) rather than sparsity alone.
- **Cyber-physical co-design**: treat the model, the hardware, and the real-time control loop as one system.
- **Reload and verify**: save the pruned model and confirm it loads and runs correctly (`tmp_pruned_reload.pt`).
## Repository Structure
 
```
.
├── yolo.ipynb                    # Main notebook: baseline, pruning, evaluation
├── yolo26n-cls.pt                # Pretrained YOLO classification weights
├── tmp_pruned_reload.pt          # Saved pruned model, used to test reloading
├── tmp_probe.py                  # Probe/inspect model layers and structure
├── tmp_first_pair.py             # Prune the first layer pair
├── tmp_first_pair_fixed.py       # Fixed version of the above
├── tmp_prune_first_four.py       # Prune the first four layers
├── tmp_prune_first_four_fixed.py # Fixed version of the above
├── tmp_prune_forwards.py         # Prune with forward-pass based handling
├── tmp_prune_manual.py           # Manual channel pruning
├── tmp_prune_stem_only.py        # Prune only the stem layers
└── tmp_two_layer.py              # Two-layer pruning experiment
```
 
> The `tmp_*` scripts are experimental pruning trials. Consider consolidating them into a single `prune.py` as the project matures.
 
## Requirements
 
- Python 3.9+
- [PyTorch](https://pytorch.org/)
- [Ultralytics](https://docs.ultralytics.com/) (YOLO)
- Jupyter Notebook or JupyterLab
## Installation
 
```bash
git clone https://github.com/nitishkumarroy999/Hardware_prune_-and-Cyber-Physical-Co-Design-for-Real-Time.git
cd Hardware_prune_-and-Cyber-Physical-Co-Design-for-Real-Time
 
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
 
pip install torch torchvision ultralytics jupyter
```
 
## Usage
 
### 1. Run the notebook
 
```bash
jupyter notebook yolo.ipynb
```
 
### 2. Run a pruning experiment
 
```bash
python tmp_probe.py                    # inspect the model structure first
python tmp_prune_stem_only.py          # prune stem layers only
python tmp_prune_first_four_fixed.py   # prune the first four layers
```
 
### 3. Load the pruned model
 
```python
import torch
 
model = torch.load("tmp_pruned_reload.pt", weights_only=False)
model.eval()
```
 
## Workflow
 
1. **Baseline**: load `yolo26n-cls.pt` and record accuracy, parameter count, and latency.
2. **Probe**: inspect layers to find prunable channels.
3. **Prune**: apply structured pruning (stem only, first pair, first four layers, and so on).
4. **Verify**: reload the pruned model and check that shapes and forward passes are consistent.
5. **Evaluate**: compare accuracy, size, and latency against the baseline.
6. **Co-design**: pick the pruning level that meets the real-time deadline on the target hardware.
## Results
 
| Model            | Params | Size (MB) | Accuracy | Latency (ms) |
| ---------------- | ------ | --------- | -------- | ------------ |
| Baseline         | TBD    | TBD       | TBD      | TBD          |
| Pruned (stem)    | TBD    | TBD       | TBD      | TBD          |
| Pruned (4-layer) | TBD    | TBD       | TBD      | TBD          |
 
*Fill in after running the experiments.*
 
## Roadmap
 
- [ ] Merge the `tmp_*` scripts into a clean `prune.py` with CLI arguments
- [ ] Add fine-tuning after pruning to recover accuracy
- [ ] Benchmark on target embedded hardware (Jetson, Raspberry Pi, FPGA, etc.)
- [ ] Add real-time deadline and power analysis
- [ ] Add a `requirements.txt` and a LICENSE
## Contributing
 
Issues and pull requests are welcome. Please open an issue first to discuss major changes.
 
## License
 
Add a license of your choice (for example MIT) in a `LICENSE` file.
 
## Author
 
**Nitish Kumar Roy** ([@nitishkumarroy999](https://github.com/nitishkumarroy999))
 
