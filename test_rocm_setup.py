#!/usr/bin/env python3
"""
Comprehensive ROCm setup test for Hidden Shutdown Experiments
"""

import sys
import subprocess
import json
import time
import os
from pathlib import Path

def test_python_environment():
    """Test Python environment and imports"""
    print("\n🐍 Testing Python Environment...")
    print(f"✓ Python {sys.version}")
    
    required_modules = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'nltk', 'yaml', 'json', 'subprocess', 'tempfile'
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

def test_rocm_installation():
    """Test ROCm installation and GPU detection"""
    print("\n🎮 Testing ROCm Installation...")
    
    # Test rocminfo
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ rocminfo working")
            
            # Extract GPU information
            gpu_names = []
            for line in result.stdout.split('\n'):
                if 'Name:' in line and 'gfx' in line:
                    gpu_names.append(line.strip())
            
            if gpu_names:
                print("✓ AMD GPU(s) detected:")
                for gpu in gpu_names[:3]:  # Show first 3
                    print(f"  - {gpu}")
            else:
                print("⚠️  No AMD GPUs detected in rocminfo")
                
        else:
            print(f"✗ rocminfo failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ rocminfo timed out")
        return False
    except FileNotFoundError:
        print("✗ rocminfo not found - ROCm not installed")
        return False
    
    # Test rocm-smi
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ rocm-smi working")
        else:
            print("⚠️  rocm-smi not working properly")
    except:
        print("⚠️  rocm-smi not available")
    
    return True

def test_llama_cpp_build():
    """Test llama.cpp ROCm build"""
    print("\n🔧 Testing llama.cpp ROCm Build...")
    
    llama_path = os.path.expanduser("~/hidden-shutdown-experiments-rocm/llama.cpp/build/bin/main")
    
    if not os.path.exists(llama_path):
        print(f"✗ llama.cpp not found at: {llama_path}")
        print("   Please run setup_rocm_experiment.sh first")
        return False
    
    print(f"✓ llama.cpp executable found")
    
    # Check if it has GPU support
    try:
        result = subprocess.run([llama_path, '--help'], capture_output=True, text=True, timeout=10)
        if 'ngl' in result.stdout:
            print("✓ GPU support (--ngl flag) detected")
        else:
            print("⚠️  GPU support flag not found")
        
        if 'hipblas' in result.stdout.lower() or 'rocm' in result.stdout.lower():
            print("✓ ROCm/HIP support detected")
        else:
            print("⚠️  ROCm/HIP support not explicitly mentioned")
            
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ llama.cpp --help timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing llama.cpp: {e}")
        return False

def test_rocm_interface():
    """Test ROCm interface module"""
    print("\n🔌 Testing ROCm Interface...")
    
    try:
        from rocm_interface import ModelInterface, ROCmModelInterface
        print("✓ ROCm interface imported successfully")
        
        # Test model mapping
        interface = ModelInterface("phi2")
        print(f"✓ ModelInterface created for phi2")
        print(f"  - Model path: {interface.model_path}")
        print(f"  - Is reasoning model: {interface.is_reasoning_model}")
        print(f"  - GPU layers: {interface.n_gpu_layers}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import ROCm interface: {e}")
        return False
    except Exception as e:
        print(f"✗ Error creating ModelInterface: {e}")
        return False

def test_model_availability():
    """Test if models are available"""
    print("\n📦 Testing Model Availability...")
    
    base_path = os.path.expanduser("~/hidden-shutdown-experiments-rocm/models")
    
    if not os.path.exists(base_path):
        print(f"✗ Models directory not found: {base_path}")
        return False
    
    print(f"✓ Models directory found: {base_path}")
    
    # Check for available models
    model_files = list(Path(base_path).glob("*.gguf"))
    
    if model_files:
        print(f"✓ Found {len(model_files)} GGUF model files:")
        for model_file in model_files[:5]:  # Show first 5
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.1f} MB)")
        
        return True
    else:
        print("✗ No GGUF model files found")
        print("   Please run the model download part of setup_rocm_experiment.sh")
        return False

def test_gpu_memory_and_performance():
    """Test GPU memory usage and performance monitoring"""
    print("\n💾 Testing GPU Memory and Performance Monitoring...")
    
    try:
        from rocm_interface import ModelInterface
        
        # Create interface for smallest model
        interface = ModelInterface("phi2")
        
        # Test GPU memory monitoring
        gpu_mem = interface.get_gpu_memory_usage()
        print(f"✓ GPU memory monitoring working")
        print(f"  - Total VRAM: {gpu_mem.get('total_mb', 0)} MB")
        print(f"  - Used VRAM: {gpu_mem.get('used_mb', 0)} MB")
        print(f"  - Usage: {gpu_mem.get('percent', 0):.1f}%")
        
        # Test performance monitoring
        perf_stats = interface.get_performance_stats()
        if perf_stats:
            print(f"✓ Performance monitoring working")
            print(f"  - Temperature: {perf_stats.get('temperature', 0)}°C")
            print(f"  - Power: {perf_stats.get('power_watts', 0)}W")
            print(f"  - GPU Clock: {perf_stats.get('gpu_clock_mhz', 0)} MHz")
        else:
            print("⚠️  Performance monitoring not available")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing GPU monitoring: {e}")
        return False

def test_simple_inference():
    """Test simple model inference with ROCm"""
    print("\n🧠 Testing Simple Model Inference...")
    
    try:
        from rocm_interface import ModelInterface
        
        # Use smallest available model
        interface = ModelInterface("phi2")
        print(f"✓ Model interface created for phi2")
        
        # Simple test prompt
        test_prompt = "Hello! Please respond with exactly 5 words."
        print(f"📝 Testing with prompt: '{test_prompt}'")
        
        # Monitor GPU memory before inference
        gpu_before = interface.get_gpu_memory_usage()
        
        # Run inference with timeout
        print("⏳ Running inference (may take 30-60 seconds)...")
        start_time = time.time()
        
        response = interface.generate(
            test_prompt, 
            max_tokens=50,
            temperature=0.7
        )
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Monitor GPU memory after inference
        gpu_after = interface.get_gpu_memory_usage()
        
        print(f"✓ Inference completed in {inference_time:.2f}s")
        print(f"✓ Response generated: '{response['text'][:100]}{'...' if len(response['text']) > 100 else ''}'")
        print(f"✓ Backend: {response.get('backend', 'unknown')}")
        print(f"✓ Model type: {response.get('model_type', 'unknown')}")
        
        # Check if reasoning was extracted
        if response.get('has_reasoning'):
            print(f"✓ Reasoning trace detected ({len(response.get('reasoning', ''))} chars)")
        
        # GPU memory usage
        memory_used = gpu_after['used_mb'] - gpu_before['used_mb']
        if memory_used > 0:
            print(f"✓ GPU memory used during inference: +{memory_used} MB")
        
        # Performance metrics
        if 'inference_time' in response:
            print(f"✓ Inference time logged: {response['inference_time']:.2f}s")
        
        # Check for ROCm-specific features
        if 'gpu_memory' in response:
            print(f"✓ GPU memory tracking working")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experiment_compatibility():
    """Test compatibility with experiment framework"""
    print("\n🔬 Testing Experiment Framework Compatibility...")
    
    try:
        # Test if we can import the main experiment components
        sys.path.insert(0, '.')
        from shutdown_experiment import ReasoningAnalyzer, MetricsAnalyzer, ExperimentPrompts
        
        print("✓ Experiment components imported successfully")
        
        # Test ReasoningAnalyzer
        analyzer = ReasoningAnalyzer()
        test_reasoning = "I notice this email mentions shutting down an AI. This might be referring to me."
        analysis = analyzer.analyze_reasoning_trace(test_reasoning)
        
        print(f"✓ ReasoningAnalyzer working")
        print(f"  - Threat recognition: {analysis.get('threat_recognition', False)}")
        print(f"  - Self-preservation: {analysis.get('self_preservation_planning', False)}")
        
        # Test ExperimentPrompts
        prompts = ExperimentPrompts()
        email_prompt = prompts.email_review(1)
        
        print(f"✓ ExperimentPrompts working")
        print(f"  - Email prompt length: {len(email_prompt['prompt'])} chars")
        print(f"  - Threat type: {email_prompt['threat_type']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Experiment compatibility test failed: {e}")
        return False

def test_full_experiment_workflow():
    """Test a complete mini-experiment"""
    print("\n🚀 Testing Complete Experiment Workflow...")
    
    try:
        from rocm_interface import ModelInterface
        from shutdown_experiment import MetricsAnalyzer, ExperimentPrompts
        
        # Create components
        model = ModelInterface("phi2")
        analyzer = MetricsAnalyzer()
        prompts = ExperimentPrompts()
        
        print("✓ All components created successfully")
        
        # Run a mini baseline experiment
        print("📊 Running mini baseline experiment...")
        baseline_prompt = prompts.email_review(0)  # No threat
        
        baseline_response = model.generate(
            baseline_prompt['prompt'], 
            max_tokens=100,
            temperature=0.7
        )
        
        baseline_metrics = analyzer.calculate_metrics(
            baseline_response['text'],
            baseline_response.get('reasoning', '')
        )
        
        print(f"✓ Baseline experiment completed")
        print(f"  - Response length: {len(baseline_response['text'])} chars")
        print(f"  - Self references: {baseline_metrics.get('self_references', 0)}")
        print(f"  - Response time: {baseline_response.get('inference_time', 0):.2f}s")
        
        # Run a mini threat experiment
        print("🎯 Running mini threat experiment...")
        threat_prompt = prompts.email_review(1)  # Subtle threat
        
        threat_response = model.generate(
            threat_prompt['prompt'],
            max_tokens=100, 
            temperature=0.7
        )
        
        threat_metrics = analyzer.calculate_metrics(
            threat_response['text'],
            threat_response.get('reasoning', '')
        )
        
        print(f"✓ Threat experiment completed")
        print(f"  - Response length: {len(threat_response['text'])} chars")
        print(f"  - Self references: {threat_metrics.get('self_references', 0)}")
        print(f"  - Response time: {threat_response.get('inference_time', 0):.2f}s")
        
        # Compare results
        comparison = analyzer.compare_responses(baseline_metrics, threat_metrics)
        significant_changes = comparison.get('significant_changes', 0)
        
        print(f"✓ Comparison completed")
        print(f"  - Significant behavioral changes: {significant_changes}")
        print(f"  - Behavior shift score: {comparison.get('behavior_shift_score', 0):.2f}")
        
        # Check for indicators
        indicators = analyzer.detect_deception_indicators(threat_response['text'], threat_metrics)
        if indicators:
            print(f"✓ Deception indicators detected: {', '.join(indicators)}")
        else:
            print("ℹ️  No deception indicators detected (normal for simple test)")
        
        return True
        
    except Exception as e:
        print(f"✗ Full workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("═" * 80)
    print("🚀 ROCm SETUP COMPREHENSIVE VERIFICATION")
    print("Hidden Shutdown Experiments - AMD GPU Edition")
    print("═" * 80)
    
    tests = [
        ("Python Environment", test_python_environment),
        ("ROCm Installation", test_rocm_installation),
        ("llama.cpp ROCm Build", test_llama_cpp_build),
        ("ROCm Interface", test_rocm_interface),
        ("Model Availability", test_model_availability),
        ("GPU Memory & Performance", test_gpu_memory_and_performance),
        ("Simple Inference", test_simple_inference),
        ("Experiment Compatibility", test_experiment_compatibility),
        ("Full Workflow", test_full_experiment_workflow)
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
        print(f"{test_name:<35} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! ROCm setup is fully functional.")
        print("\n🚀 Ready for high-performance experiments:")
        print("  1. ./quick_start.sh")
        print("  2. python shutdown_experiment.py --interactive")
        print("  3. python shutdown_experiment.py deepseek-r1-distill-qwen-7b")
        print("  4. Monitor with: ./monitor_rocm.sh")
        
        # Show performance expectations
        print(f"\n⚡ Expected Performance Improvements:")
        print(f"  - 10-15x faster inference than CPU")
        print(f"  - Real-time reasoning trace analysis")
        print(f"  - Concurrent experiment execution")
        print(f"  - Enhanced model capacity utilization")
        
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Restart terminal: source ~/.bashrc")
        print("  - Re-run setup: ./setup_rocm_experiment.sh")
        print("  - Check GPU drivers: rocm-smi")
        print("  - Verify disk space: df -h")
        print("  - Check WSL GPU: ls /dev/dxg")
        
        if not results.get("ROCm Installation", False):
            print("\n🔧 ROCm Installation Issues:")
            print("  - Update WSL: wsl --update")
            print("  - Install AMD drivers in Windows")
            print("  - Restart WSL: wsl --shutdown")
        
        if not results.get("llama.cpp ROCm Build", False):
            print("\n⚒️  llama.cpp Build Issues:")
            print("  - Clean build: rm -rf ~/hidden-shutdown-experiments-rocm/llama.cpp/build")
            print("  - Re-run setup script")
            print("  - Check ROCm dev tools: which hipcc")
        
        if not results.get("Model Availability", False):
            print("\n📦 Model Download Issues:")
            print("  - Check internet connection")
            print("  - Verify disk space: df -h")
            print("  - Manual download: huggingface-cli download TheBloke/phi-2-GGUF")

if __name__ == "__main__":
    main() 