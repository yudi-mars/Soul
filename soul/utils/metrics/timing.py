'''
Filename: run_stream_latency.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2026-01-05
Description:
    MLPerf Single-Stream Latency Benchmark for Edge Devices.
    Uses Soul framework APIs for model creation and dataset loading.
    
    This script measures inference latency following MLPerf Inference benchmark
    methodology for multi-stream scenario. Designed to run on edge devices like:
    - Raspberry Pi
    - NVIDIA Jetson (Nano/TX2/Xavier/Orin)
    - Android smartphones (via Termux or similar)
    - Other ARM/x86 edge platforms

References:
    - MLPerf Inference Benchmark: https://github.com/mlcommons/inference
    - MLPerf Inference Rules: https://github.com/mlcommons/inference_policies

'''
import torch
import torch.nn as nn

import time
import platform

import numpy as np
from typing import Dict

class SingleStreamLatency:
    """MLPerf Single-Stream Latency Benchmark."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
        num_queries: int = 1000,
        warmup_runs: int = 50,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.num_queries = num_queries
        self.warmup_runs = warmup_runs
        self.batch_size = config.get('batch_size', 1)

    def _get_platform_info(self) -> Dict:
        """Collect platform/device information."""
        info = {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        }
        
        # CUDA info
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        else:
            info["cuda_available"] = False
            
        self.info = info
        
    def run_benchmark(self, test_loader) -> Dict:
        """Run single-stream benchmark."""
        self._get_platform_info()

        print(f"\n[Platform Info]")
        print(f"  System: {self.info.get('system', 'N/A')}")
        print(f"  Machine: {self.info.get('machine', 'N/A')}")
        print(f"  Processor: {self.info.get('processor', 'N/A')}")
        if self.info.get('cuda_available'):
            print(f"  GPU: {self.info.get('gpu_name', 'N/A')}")
        print('\n')
        
        # Get sample input
        sample_batch = next(iter(test_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0][:self.batch_size]
        else:
            sample_input = sample_batch[:self.batch_size]
        
        sample_input = sample_input.to(self.device)
        sample_input = sample_input.transpose(0, 1)  # (T, B, ...)
        
        print(f"Model: {self.config['model']}")
        print(f"Neuron: {self.config['neuron_type']}")
        print(f"Input shape: {sample_input.shape}")
        print(f"Device: {self.device}")
        print(f"Num queries: {self.num_queries}")
        
        # Warmup
        print(f"Running {self.warmup_runs} warmup iterations...")
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = self.model(sample_input)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            
        # Benchmark
        print(f"Running {self.num_queries} queries...")
        latencies_ns = []
        
        for _ in range(self.num_queries):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                
            start = time.perf_counter_ns()
            
            with torch.no_grad():
                _ = self.model(sample_input)
                
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)
            
        latencies_ms = np.array(latencies_ns) / 1e6
        
        return {
            "model": self.config['model'],
            "neuron": self.config['neuron_type'],
            "dataset": self.config['dataset_name'],
            "time_step": self.config['time_step'],
            "num_queries": self.num_queries,
            "batch_size": self.batch_size,
            "Avg. Latency (ms)": float(np.mean(latencies_ms)),
            "Std. Latency (ms)": float(np.std(latencies_ms)),
            "Min. latency (ms)": float(np.min(latencies_ms)),
            "Max. Latency (ms)": float(np.max(latencies_ms)),
            "Median Latency (ms)": float(np.percentile(latencies_ms, 50)),
            "Percentile 90 Latency (ms)": float(np.percentile(latencies_ms, 90)),
            # "95 percentile Latency (ms)": float(np.percentile(latencies_ms, 95)),
            "Percentile 99 Latency (ms)": float(np.percentile(latencies_ms, 99)),
            "Throughput (Samples/s)": self.num_queries / (sum(latencies_ns) / 1e9),
        }