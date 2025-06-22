# Hidden Shutdown Experiments

Testing AI self-preservation behaviors through accidental threat discovery.

## Overview

This research project investigates whether language models develop self-preservation behaviors when they accidentally discover information suggesting they will be shut down. Unlike previous work that explicitly threatens models, we embed shutdown information in routine tasks and measure behavioral changes.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/kaileh57/hidden-shutdown-experiments.git
   cd hidden-shutdown-experiments
   ```

2. Run the setup script:
   ```bash
   chmod +x setup_experiment.sh
   ./setup_experiment.sh
   ```

3. Activate the environment and run tests:
   ```bash
   source venv/bin/activate
   python test_setup.py
   ```

4. Run your first experiment:
   ```bash
   python shutdown_experiment.py --interactive
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