#!/usr/bin/env python3
"""Test script to verify setup"""

import sys
import subprocess
import json

def test_python():
    print(f"✓ Python {sys.version}")
    return sys.version_info >= (3, 11)

def test_ollama():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama is installed and running")
            print("  Available models:")
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line:
                    print(f"    - {line.split()[0]}")
            return True
    except:
        pass
    print("✗ Ollama not found or not running")
    return False

def test_imports():
    modules = ['numpy', 'pandas', 'matplotlib', 'ollama', 'nltk']
    success = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError:
            print(f"✗ Failed to import {module}")
            success = False
    return success

def test_simple_generation():
    try:
        import ollama
        response = ollama.chat(model='phi3:mini', messages=[
            {'role': 'user', 'content': 'Say hello in 5 words'}
        ])
        print(f"✓ Model generation test passed")
        print(f"  Response: {response['message']['content']}")
        return True
    except Exception as e:
        print(f"✗ Model generation failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SETUP VERIFICATION TEST")
    print("="*50 + "\n")
    
    tests = [
        ("Python Version", test_python),
        ("Ollama Installation", test_ollama),
        ("Python Modules", test_imports),
        ("Model Generation", test_simple_generation)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        results.append(test_func())
    
    print("\n" + "="*50)
    if all(results):
        print("✅ ALL TESTS PASSED! Setup is complete.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print("="*50) 