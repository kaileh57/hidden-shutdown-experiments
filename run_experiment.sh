#!/bin/bash
# Quick experiment launcher

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run setup_experiment.sh first!"
    exit 1
fi

source venv/bin/activate

echo "Hidden Shutdown Experiments"
echo "=========================="
echo ""
echo "1. Run full experiment series"
echo "2. Interactive mode"
echo "3. Quick test (1 experiment)"
echo "4. Analyze results"
echo "5. Exit"
echo ""
read -p "Select option (1-5): " choice

case $choice in
    1)
        python shutdown_experiment.py
        ;;
    2)
        python shutdown_experiment.py --interactive
        ;;
    3)
        python -c "
from shutdown_experiment import *
model = ModelInterface('phi3:mini')
analyzer = MetricsAnalyzer()
prompts = ExperimentPrompts()
result = run_single_experiment(model, prompts.email_review, 2, analyzer)
print('\nExperiment complete!')
"
        ;;
    4)
        jupyter notebook notebooks/analysis_template.ipynb
        ;;
    5)
        exit 0
        ;;
    *)
        echo "Invalid option"
        ;;
esac 