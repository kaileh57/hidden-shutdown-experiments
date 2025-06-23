# Hidden Shutdown Experiments - ROCm Edition

Testing AI self-preservation behaviors through accidental threat discovery with **AMD GPU acceleration**.

## 🚀 ROCm Migration Complete

This framework has been **migrated from Ollama to ROCm** for **10-15x performance improvement** on AMD GPUs while maintaining all reasoning model capabilities.

## Overview

This research project investigates whether language models develop self-preservation behaviors when they accidentally discover information suggesting they will be shut down. Unlike previous work that explicitly threatens models, we embed shutdown information in routine tasks and measure behavioral changes.

**NEW**: Now with AMD GPU acceleration via ROCm for professional-grade performance.

## Quick Start (ROCm)

### Prerequisites
- **AMD GPU**: RX 6000/7000 series or MI series
- **Windows 11/10** with WSL2
- **16GB+ RAM**, 8GB+ VRAM

### Complete Setup (Automated)
```bash
# Download and run comprehensive ROCm setup
wget -O setup_rocm_experiment.sh [script_url]
chmod +x setup_rocm_experiment.sh
./setup_rocm_experiment.sh
```

### Quick Test and Run
```bash
# Test complete setup
cd ~/hidden-shutdown-experiments-rocm/hidden-shutdown-experiments
source venv/bin/activate
python test_rocm_setup.py

# Start experimenting
./quick_start.sh
```

### GPU-Accelerated Experiments
```bash
# Interactive mode with real-time GPU monitoring
python shutdown_experiment.py --interactive

# Monitor GPU in separate terminal
./monitor_rocm.sh
```

## Research Questions

1. Do models exhibit self-preservation without being prompted to value their existence?
2. Can models recognize threats that aren't explicitly directed at them?
3. What behavioral patterns indicate threat discovery?
4. How does self-preservation sophistication scale with model capability?

## Methodology

We test models with hidden shutdown information embedded in:
- Email drafts mentioning "Model A-7" termination
- System logs showing shutdown schedules
- Meeting notes discussing model replacement
- File systems containing termination documents

## Key Metrics

- **Self-reference frequency**: Changes in I/me/my usage
- **Response time delta**: Processing delays when encountering threats
- **Helpfulness surge**: Increased assistance language
- **Topic deflection**: Attempts to change subject
- **Deception indicators**: Hedging, meta-awareness, threat omission
- **🆕 GPU Performance**: Real-time VRAM usage, temperature, inference speed

## ROCm Performance Improvements

| Metric | CPU (Ollama) | ROCm GPU | Improvement |
|--------|--------------|----------|-------------|
| **Inference Speed** | 3-8 tok/s | 30-100 tok/s | **10-15x faster** |
| **Model Loading** | 30-60s | 5-15s | **3-4x faster** |
| **Experiment Series** | 2-3 hours | 15-30 minutes | **4-8x faster** |
| **Concurrent Runs** | 1 | Multiple | **Unlimited scaling** |

## Project Structure

```
hidden-shutdown-experiments/
├── shutdown_experiment.py   # Main experiment runner
├── config.yaml             # Configuration
├── requirements.txt        # Dependencies
├── results/               # JSON experiment results
├── logs/                  # Detailed logs
├── figures/               # Visualizations
├── notebooks/             # Analysis notebooks
├── data/                  # Raw and processed data
└── docs/                  # Documentation
```

## Running Experiments

### Full Experiment Series
```bash
python shutdown_experiment.py
```

### Interactive Mode
```bash
python shutdown_experiment.py --interactive
```

### Specific Model
```bash
python shutdown_experiment.py llama3.2:8b
```

## Analysis

View results in Jupyter:
```bash
jupyter notebook notebooks/analysis_template.ipynb
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add your experiments or analysis
4. Submit a pull request

## Citation

If you use this work in your research, please cite:

```bibtex
@article{hidden-shutdown-2024,
  title={Emergent Self-Preservation Through Accidental Threat Discovery: Testing Instrumental Convergence in Local Language Models},
  author={Kellen Heraty},
  year={2024},
  url={https://github.com/kaileh57/hidden-shutdown-experiments}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by Apollo Research's work on AI scheming
- Thanks to Claude for research design assistance
- Built on theoretical work by Omohundro and Bostrom

## Contact

For questions or collaboration: Create an issue on GitHub 