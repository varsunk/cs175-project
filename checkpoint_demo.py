#!/usr/bin/env python3
"""
Demonstration script for checkpoint loading functionality in the Malmo DQN training.

Usage examples:
1. Start fresh training (ignoring checkpoints): python run.py --no-checkpoint
2. Load latest checkpoint automatically: python run.py
3. Load best checkpoint: python run.py --load-checkpoint best
4. Load specific checkpoint: python run.py --load-checkpoint /path/to/checkpoint.pth

This script shows how to check what checkpoints are available.
"""

import os
import torch
import json
from pathlib import Path

def check_checkpoints(models_dir="models", metrics_dir="metrics"):
    """Check what checkpoints are available and display their info"""
    
    print("🔍 CHECKPOINT SCANNER")
    print("=" * 50)
    
    # Check if directories exist
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found.")
        print("No checkpoints available. Training will start from scratch.")
        return
    
    if not os.path.exists(metrics_dir):
        print(f"❌ Metrics directory '{metrics_dir}' not found.")
        print("No metrics available.")
        return
    
    # Check for latest checkpoint
    latest_model = os.path.join(models_dir, "latest_checkpoint.pth")
    latest_metrics = os.path.join(metrics_dir, "latest_metrics.json")
    
    if os.path.exists(latest_model):
        print("✅ LATEST CHECKPOINT FOUND")
        try:
            checkpoint = torch.load(latest_model, map_location='cpu')
            print(f"   📊 Episode: {checkpoint['episode']}")
            print(f"   🎯 Epsilon: {checkpoint['epsilon']:.3f}")
            print(f"   📈 Last Reward: {checkpoint['reward']:.2f}")
            print(f"   🏆 Best Reward: {checkpoint['best_reward']:.2f}")
            
            if os.path.exists(latest_metrics):
                with open(latest_metrics, 'r') as f:
                    metrics = json.load(f)
                    total_episodes = len(metrics.get('rewards', []))
                    completions = len(metrics.get('completion_times', []))
                    print(f"   📋 Total Episodes Trained: {total_episodes}")
                    print(f"   🏁 Successful Completions: {completions}")
                    
                    if completions > 0:
                        avg_completion_time = sum(c['time'] for c in metrics['completion_times']) / completions
                        best_time = min(c['time'] for c in metrics['completion_times'])
                        print(f"   ⏱️  Average Completion Time: {avg_completion_time:.2f}s")
                        print(f"   🥇 Best Completion Time: {best_time:.2f}s")
        except Exception as e:
            print(f"   ❌ Error reading checkpoint: {e}")
        print()
    else:
        print("❌ No latest checkpoint found")
        print()
    
    # Check for best checkpoint
    best_model = os.path.join(models_dir, "best_checkpoint.pth")
    best_metrics = os.path.join(metrics_dir, "best_metrics.json")
    
    if os.path.exists(best_model):
        print("🏆 BEST CHECKPOINT FOUND")
        try:
            checkpoint = torch.load(best_model, map_location='cpu')
            print(f"   📊 Episode: {checkpoint['episode']}")
            print(f"   🎯 Epsilon: {checkpoint['epsilon']:.3f}")
            print(f"   📈 Reward: {checkpoint['reward']:.2f}")
            print(f"   🏆 Best Reward: {checkpoint['best_reward']:.2f}")
        except Exception as e:
            print(f"   ❌ Error reading checkpoint: {e}")
        print()
    else:
        print("❌ No best checkpoint found")
        print()
    
    # Check for completion times file
    completion_file = os.path.join(metrics_dir, "completion_times.txt")
    if os.path.exists(completion_file):
        print("📄 COMPLETION TIMES REPORT AVAILABLE")
        print(f"   📁 File: {completion_file}")
        
        # Show last few lines of the report
        try:
            with open(completion_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 10:
                    print("   📋 Recent entries:")
                    for line in lines[-10:]:
                        if line.strip():
                            print(f"      {line.strip()}")
        except Exception as e:
            print(f"   ❌ Error reading completion times: {e}")
    else:
        print("❌ No completion times report found")
    
    print("=" * 50)
    print("\n💡 USAGE TIPS:")
    print("• To resume from latest checkpoint: python run.py")
    print("• To start fresh: python run.py --no-checkpoint")
    print("• To load best checkpoint: python run.py --load-checkpoint best")
    print("• To load specific file: python run.py --load-checkpoint /path/to/file.pth")

if __name__ == "__main__":
    check_checkpoints() 