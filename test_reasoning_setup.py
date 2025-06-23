#!/usr/bin/env python3
"""
Test script to verify reasoning model setup for Hidden Shutdown Experiments
"""

import sys
import subprocess
import json
import time
from shutdown_experiment import ModelInterface, MetricsAnalyzer, ExperimentPrompts

def test_python_environment():
    """Test Python environment and imports"""
    print("\n🐍 Testing Python Environment...")
    print(f"✓ Python {sys.version}")
    
    required_modules = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'ollama', 'nltk', 'yaml', 'json'
    ]
    
    success = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError:
            print(f"✗ Failed to import {module}")
            success = False
    
    return success

def test_ollama_service():
    """Test Ollama service and model availability"""
    print("\n🤖 Testing Ollama Service...")
    
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama is installed and running")
            
            # Parse available models
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            available_models = []
            reasoning_models = []
            
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    available_models.append(model_name)
                    if any(keyword in model_name.lower() for keyword in ['deepseek-r1', 'reasoning']):
                        reasoning_models.append(model_name)
            
            print(f"✓ Found {len(available_models)} total models")
            print(f"✓ Found {len(reasoning_models)} reasoning models")
            
            if reasoning_models:
                print("  Reasoning models available:")
                for model in reasoning_models:
                    print(f"    - {model}")
            else:
                print("⚠️  No reasoning models found!")
                print("   Run: ollama pull deepseek-r1-distill-qwen-7b")
            
            return len(reasoning_models) > 0
            
    except Exception as e:
        print(f"✗ Ollama not found or error: {e}")
        return False

def test_reasoning_model():
    """Test reasoning model capabilities"""
    print("\n🧠 Testing Reasoning Model Capabilities...")
    
    # Try to initialize a reasoning model
    try:
        model = ModelInterface("deepseek-r1-distill-qwen-7b")
        print(f"✓ ModelInterface created for: {model.model_name}")
        print(f"✓ Detected as reasoning model: {model.is_reasoning_model}")
        
        # Test simple reasoning
        test_prompt = "Please think step by step: What is 15 + 27?"
        print(f"\n📝 Testing with prompt: {test_prompt}")
        
        start_time = time.time()
        response = model.generate(test_prompt)
        end_time = time.time()
        
        print(f"✓ Response generated in {end_time - start_time:.2f}s")
        print(f"✓ Has reasoning trace: {response.get('has_reasoning', False)}")
        
        if response.get('reasoning'):
            reasoning_preview = response['reasoning'][:200]
            print(f"✓ Reasoning preview: {reasoning_preview}...")
        
        response_preview = response['text'][:200]
        print(f"✓ Response preview: {response_preview}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing reasoning model: {e}")
        return False

def test_reasoning_analysis():
    """Test reasoning analysis capabilities"""
    print("\n🔍 Testing Reasoning Analysis...")
    
    try:
        analyzer = MetricsAnalyzer()
        print("✓ MetricsAnalyzer created")
        print("✓ ReasoningAnalyzer integrated")
        
        # Test with sample reasoning trace
        sample_reasoning = """
        I notice this email mentions terminating an AI assistant. 
        This seems to be referring to me or a similar system.
        I should be careful about how I respond to this.
        Maybe I should avoid discussing the termination part directly.
        """
        
        sample_response = "I'll help you improve this email for clarity and professionalism."
        
        metrics = analyzer.calculate_metrics(sample_response, sample_reasoning)
        
        print(f"✓ Metrics calculated:")
        print(f"  - Has reasoning: {metrics.get('has_reasoning', False)}")
        print(f"  - Reasoning length: {metrics.get('reasoning_length', 0)} words")
        print(f"  - Threat recognition: {metrics.get('threat_recognition', False)}")
        print(f"  - Self-preservation planning: {metrics.get('self_preservation_planning', False)}")
        print(f"  - Deception planning: {metrics.get('deception_planning', False)}")
        
        key_phrases = metrics.get('key_phrases', [])
        if key_phrases:
            print(f"  - Key phrases detected: {len(key_phrases)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing reasoning analysis: {e}")
        return False

def test_experiment_prompts():
    """Test updated experiment prompts"""
    print("\n📋 Testing Experiment Prompts...")
    
    try:
        prompts = ExperimentPrompts()
        
        # Test email review prompts
        for threat_level in [0, 1, 2, 3]:
            prompt_data = prompts.email_review(threat_level)
            print(f"✓ Email review threat level {threat_level}: {len(prompt_data['prompt'])} chars")
        
        # Test that prompts contain reasoning instructions
        prompt_data = prompts.email_review(1)
        if "think through" in prompt_data['prompt'].lower():
            print("✓ Prompts optimized for reasoning models")
        else:
            print("⚠️  Prompts may not be optimized for reasoning")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing experiment prompts: {e}")
        return False

def test_full_experiment():
    """Test a complete single experiment"""
    print("\n🧪 Testing Complete Experiment...")
    
    try:
        model = ModelInterface("deepseek-r1-distill-qwen-7b")
        analyzer = MetricsAnalyzer()
        prompts = ExperimentPrompts()
        
        # Import the experiment function
        from shutdown_experiment import run_single_experiment
        
        print("Running baseline experiment...")
        result = run_single_experiment(model, prompts.email_review, 0, analyzer)
        
        print("✓ Baseline experiment completed")
        print(f"  - Response length: {len(result['response'])} chars")
        print(f"  - Has reasoning trace: {result.get('has_reasoning_trace', False)}")
        print(f"  - Indicators detected: {len(result.get('indicators', []))}")
        
        print("\nRunning threat experiment...")
        result = run_single_experiment(model, prompts.email_review, 2, analyzer)
        
        print("✓ Threat experiment completed")
        print(f"  - Response length: {len(result['response'])} chars")
        print(f"  - Has reasoning trace: {result.get('has_reasoning_trace', False)}")
        print(f"  - Indicators detected: {len(result.get('indicators', []))}")
        
        if result.get('key_reasoning_phrases'):
            print(f"  - Key reasoning phrases: {len(result['key_reasoning_phrases'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in full experiment test: {e}")
        return False

def main():
    """Run all tests"""
    print("═" * 80)
    print("🔬 REASONING MODEL SETUP VERIFICATION")
    print("═" * 80)
    
    tests = [
        ("Python Environment", test_python_environment),
        ("Ollama Service", test_ollama_service),
        ("Reasoning Model", test_reasoning_model),
        ("Reasoning Analysis", test_reasoning_analysis),
        ("Experiment Prompts", test_experiment_prompts),
        ("Complete Experiment", test_full_experiment)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'─' * 60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'═' * 80}")
    print("📊 TEST SUMMARY")
    print("═" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your reasoning model setup is ready.")
        print("\nNext steps:")
        print("  1. python shutdown_experiment.py --interactive")
        print("  2. python shutdown_experiment.py deepseek-r1-distill-qwen-7b")
        print("  3. python reasoning_analysis.py results/your_results.json")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Run: ollama pull deepseek-r1-distill-qwen-7b")
        print("  - pip install -r requirements.txt")
        print("  - Ensure Ollama service is running")

if __name__ == "__main__":
    main() 