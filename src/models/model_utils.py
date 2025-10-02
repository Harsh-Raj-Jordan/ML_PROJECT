"""
Utility functions for model management and evaluation
"""

import torch
import json
from pathlib import Path
from datetime import datetime

class ModelManager:
    """Manage multiple translation models"""
    
    def __init__(self):
        self.models = {}
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    def register_model(self, name, model_instance):
        """Register a model instance"""
        self.models[name] = model_instance
        print(f"✅ Registered model: {name}")
    
    def get_model(self, name):
        """Get a model by name"""
        return self.models.get(name)
    
    def list_models(self):
        """List all registered models"""
        return list(self.models.keys())
    
    def save_model_info(self):
        """Save model information to file"""
        model_info = {}
        for name, model in self.models.items():
            if hasattr(model, 'get_model_info'):
                model_info[name] = model.get_model_info()
        
        info_file = self.model_dir / "model_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Model info saved to: {info_file}")

def check_gpu_availability():
    """Check GPU availability and info"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            gpu_info.append(f"GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
        
        return True, gpu_info
    else:
        return False, ["No GPU available - using CPU"]

def benchmark_translation(model, sentences, repetitions=3):
    """Benchmark translation speed"""
    import time
    
    print(f"⏱️  Benchmarking translation speed...")
    
    times = []
    for _ in range(repetitions):
        start_time = time.time()
        _ = model.batch_translate(sentences)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    speed = len(sentences) / avg_time  # sentences per second
    
    print(f"📊 Benchmark Results:")
    print(f"   Sentences: {len(sentences)}")
    print(f"   Average time: {avg_time:.2f}s")
    print(f"   Speed: {speed:.1f} sentences/second")
    
    return speed