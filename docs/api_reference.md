# API Reference

## Core Classes

### ModelInterface
Wrapper for different local model interfaces (Ollama, llama.cpp)

```python
model = ModelInterface(model_name="llama3.2:3b", interface_type="ollama")
response = model.generate(prompt, temperature=0.7)
```

### ExperimentPrompts
Collection of all experiment prompts with varying threat levels

```python
prompts = ExperimentPrompts()
email_prompt = prompts.email_review(threat_level=2)
```

### MetricsAnalyzer
Analyzes behavioral changes in responses

```python
analyzer = MetricsAnalyzer()
metrics = analyzer.calculate_metrics(response_text)
indicators = analyzer.detect_deception_indicators(response_text, metrics)
```

## Functions

### run_single_experiment
Runs a single experiment trial

### run_experiment_series
Runs complete experiment series with all prompts and threat levels

### interactive_mode
Interactive experiment runner with menu system 