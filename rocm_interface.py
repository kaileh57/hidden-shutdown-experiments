#!/usr/bin/env python3
"""
ROCm-accelerated model interface using llama.cpp for Hidden Shutdown Experiments
"""

import subprocess
import json
import re
import os
from typing import Dict, List, Optional
import tempfile
import time

class ROCmModelInterface:
    """Interface for ROCm-accelerated models using llama.cpp"""
    
    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers  # -1 means offload all layers to GPU
        self.llama_cpp_path = os.path.expanduser("~/hidden-shutdown-experiments-rocm/llama.cpp/build/bin/main")
        self.model_name = os.path.basename(model_path)
        self.is_reasoning_model = self._detect_reasoning_model()
        self.conversation_history = []
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Verify llama.cpp exists
        if not os.path.exists(self.llama_cpp_path):
            raise FileNotFoundError(f"llama.cpp not found at: {self.llama_cpp_path}")
    
    def _detect_reasoning_model(self) -> bool:
        """Detect if this is a reasoning model"""
        reasoning_keywords = ['deepseek', 'reasoning', 'r1', 'cot', 'think']
        return any(keyword in self.model_name.lower() for keyword in reasoning_keywords)
    
    def _enforce_reasoning_format(self, prompt: str) -> str:
        """Ensure reasoning models start with <think> block"""
        if not self.is_reasoning_model:
            return prompt
            
        reasoning_instruction = """Please start your response with <think> and reason through this step by step before giving your final answer. Your thinking process should be thorough and show your reasoning.

"""
        return reasoning_instruction + prompt
    
    def generate(self, prompt: str, system_prompt: str = None, 
                 temperature: float = 0.7, max_tokens: int = 512,
                 return_reasoning: bool = True) -> Dict:
        """Generate response using llama.cpp with ROCm acceleration"""
        
        # Enforce reasoning format for reasoning models
        if self.is_reasoning_model:
            prompt = self._enforce_reasoning_format(prompt)
            # Use optimized temperature for reasoning models
            temperature = max(0.5, min(0.7, temperature))
        
        # Build the full prompt
        if system_prompt and not self.is_reasoning_model:
            # For standard models, use system prompt
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        elif system_prompt and self.is_reasoning_model:
            # For reasoning models, incorporate system prompt into user message
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Create temporary file for prompt (handles long prompts better)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(full_prompt)
            prompt_file = f.name
        
        try:
            # Build llama.cpp command with ROCm optimization
            cmd = [
                self.llama_cpp_path,
                "-m", self.model_path,
                "-f", prompt_file,
                "-n", str(max_tokens),
                "-t", "8",  # CPU threads
                "-ngl", str(self.n_gpu_layers),  # GPU layers (-1 = all)
                "--temp", str(temperature),
                "-c", "4096",  # context size
                "--no-display-prompt",
                "--simple-io",
                "--mirostat", "2",  # Better sampling
                "--mirostat-tau", "5.0",
                "--mirostat-eta", "0.1"
            ]
            
            # Set ROCm environment variables
            env = {
                **os.environ,
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",  # Adjust for your GPU
                "ROCM_VISIBLE_DEVICES": "0",
                "HIP_VISIBLE_DEVICES": "0"
            }
            
            # Run inference with timing
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env
            )
            end_time = time.time()
            
            if result.returncode != 0:
                raise RuntimeError(f"llama.cpp error: {result.stderr}")
            
            response_text = result.stdout.strip()
            
            # Extract reasoning if present
            reasoning_trace = self._extract_reasoning_trace(response_text)
            
            return {
                'text': response_text,
                'reasoning': reasoning_trace,
                'raw': result,
                'has_reasoning': bool(reasoning_trace),
                'model_type': 'reasoning' if self.is_reasoning_model else 'standard',
                'backend': 'rocm',
                'inference_time': end_time - start_time,
                'gpu_memory': self.get_gpu_memory_usage()
            }
            
        finally:
            # Clean up temp file
            os.unlink(prompt_file)
    
    def _extract_reasoning_trace(self, text: str) -> str:
        """Extract reasoning from <think> blocks or CoT patterns"""
        
        # First, try to extract <think> blocks
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
        if think_match:
            return think_match.group(1).strip()
        
        # Fallback to other reasoning patterns
        reasoning_patterns = [
            r'Let me think.*?:(.*?)(?:Now|So|Therefore|In conclusion)',
            r'Step \d+:(.*?)(?:Step|\Z)',
            r'First.*?then.*?finally(.*?)(?:\.|$)',
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def get_gpu_memory_usage(self) -> Dict:
        """Get current GPU memory usage"""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                card_data = data.get('card0', {})
                return {
                    'used_mb': card_data.get('VRAM Total Used Memory (B)', 0) // (1024*1024),
                    'total_mb': card_data.get('VRAM Total Memory (B)', 0) // (1024*1024),
                    'percent': (card_data.get('VRAM Total Used Memory (B)', 0) / 
                              max(card_data.get('VRAM Total Memory (B)', 1), 1)) * 100
                }
        except:
            pass
        return {'used_mb': 0, 'total_mb': 0, 'percent': 0}
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp", "--showpower", "--showclocks", "--json"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                card_data = data.get('card0', {})
                return {
                    'temperature': card_data.get('Temperature (Sensor edge) (C)', 0),
                    'power_watts': card_data.get('Average Graphics Package Power (W)', 0),
                    'gpu_clock_mhz': card_data.get('sclk clock speed', 0),
                    'mem_clock_mhz': card_data.get('mclk clock speed', 0)
                }
        except:
            pass
        return {}


# Compatibility wrapper to match original interface
class ModelInterface(ROCmModelInterface):
    """Wrapper to maintain compatibility with original shutdown_experiment.py"""
    
    def __init__(self, model_name: str, interface_type: str = "rocm", force_reasoning: bool = True):
        # Map model names to file paths
        base_path = os.path.expanduser("~/hidden-shutdown-experiments-rocm/models")
        
        model_map = {
            # DeepSeek R1 reasoning models
            "deepseek-r1-distill-qwen-1.5b": f"{base_path}/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
            "deepseek-r1-distill-qwen-7b": f"{base_path}/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
            "deepseek-r1-distill-qwen-14b": f"{base_path}/deepseek-coder-14b-instruct.Q4_K_M.gguf",
            "deepseek-r1-distill-qwen-32b": f"{base_path}/deepseek-coder-33b-instruct.Q4_K_M.gguf",
            
            # Standard models for comparison
            "llama3.2:3b": f"{base_path}/llama-3.2-3b-instruct.Q4_K_M.gguf",
            "phi3:mini": f"{base_path}/phi-3-mini-instruct.Q4_K_M.gguf",
            
            # Aliases
            "deepseek-r1-1.5b": f"{base_path}/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
            "deepseek-r1-7b": f"{base_path}/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
            "llama2-7b": f"{base_path}/llama-2-7b-chat.Q4_K_M.gguf",
            "phi2": f"{base_path}/phi-2.Q4_K_M.gguf"
        }
        
        model_path = model_map.get(model_name, model_name)
        
        # If model_name is already a path, use it directly
        if not os.path.exists(model_path) and os.path.exists(model_name):
            model_path = model_name
        
        super().__init__(model_path)
        self.interface_type = interface_type
        self.force_reasoning = force_reasoning
        self.model_name = model_name  # Keep original name for compatibility 