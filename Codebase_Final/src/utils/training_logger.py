"""
train_logger.py — Unified logging for all model training runs.

Maintains 3 separate log files (one per model) that capture:
  - Epoch information (manifest loaded, breakdown, model specs)
  - Batch details (batch losses, statistics)
  - Epoch summaries (totals, pruning, timing)

Usage:
  logger = TrainingLogger(model_name='resnet')  # resnet, convnext, or vit
  
  # Log epoch header
  logger.log_epoch_header(manifest, model, device, epoch, num_epochs)
  
  # Log batch progress
  logger.log_batch(batch_idx, total_loss, global_loss, local_loss)
  
  # Log epoch summary
  logger.log_epoch_summary(epoch, avg_total, avg_global, avg_local, 
                           pairs_trained, pruned, remaining, type_counts, elapsed, ckpt_path)
"""

import os
from datetime import datetime


class TrainingLogger:
    def __init__(self, model_name='resnet'):
        """
        Initialize logger for a specific model.
        
        Args:
            model_name: 'resnet', 'convnext', or 'vit'
        """
        self.model_name = model_name.lower()
        self.log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, f'training_log_{self.model_name}.txt')
        self.current_epoch = 0
        self.log_batches = os.environ.get("BTP_LOG_BATCHES", "1") == "1"
    
    def _write(self, text):
        """Write to log file and append (don't overwrite)."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    
    def log_session_start(self, model_name_display):
        """Log start of training session."""
        separator = "="*70
        self._write(separator)
        self._write(f"TRAINING SESSION: {model_name_display}")
        self._write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(separator)
        self._write("")
    
    def log_epoch_header(self, manifest, model, device, epoch, num_epochs, model_display_name):
        """
        Log epoch header with manifest and model information.
        
        Args:
            manifest: List of training pairs
            model: PyTorch model
            device: 'cpu' or 'cuda'
            epoch: Current epoch number
            num_epochs: Total epochs
            model_display_name: Display name (e.g., "Siamese ResNet-50")
        """
        # Count pairs by type
        type_counts = {}
        for p in manifest:
            ptype = p.get('pair_type', 'unknown')
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        self.current_epoch = epoch
        
        # Log header
        separator = "─" * 70
        self._write(separator)
        self._write(f"EPOCH {epoch}/{num_epochs} | Pairs: {len(manifest):,}")
        self._write(separator)
        self._write(f"📋 Manifest loaded: {len(manifest):,} pairs")
        self._write(f"   Breakdown: {type_counts}")
        self._write(f"🧠 Model: {model_display_name} on {device}")
        self._write(f"   Total parameters: {total_params:,}")
        self._write("")
    
    def log_batch(self, batch_idx, total_loss, global_loss, local_loss):
        """
        Log individual batch training progress.
        
        Args:
            batch_idx: Batch number (1-indexed)
            total_loss: Total loss value
            global_loss: Global branch loss
            local_loss: Local branch loss
        """
        if not self.log_batches:
            return
        self._write(f"  Batch {batch_idx:>5} | Loss: {total_loss:.4f} | Global: {global_loss:.4f} | Local: {local_loss:.4f}")
    
    def log_epoch_summary(self, epoch, avg_total, avg_global, avg_local, 
                         pairs_trained, pruned, remaining, type_counts, elapsed, ckpt_path):
        """
        Log epoch summary after all batches complete.
        
        Args:
            epoch: Epoch number
            avg_total: Average total loss
            avg_global: Average global loss
            avg_local: Average local loss
            pairs_trained: Number of pairs trained in epoch
            pruned: Number of pairs pruned
            remaining: Remaining pairs in manifest
            type_counts: Dict of pair counts by type
            elapsed: Seconds elapsed
            ckpt_path: Path to saved checkpoint
        """
        self._write("")
        self._write(f"  📊 Epoch {epoch} Summary:")
        self._write(f"     Loss: {avg_total:.4f}  (Global: {avg_global:.4f}, Local: {avg_local:.4f})")
        self._write(f"     Pairs trained: {pairs_trained:,}")
        self._write(f"     Pruned: {pruned:,}")
        self._write(f"     Remaining: {remaining:,}")
        self._write(f"     By type: {type_counts}")
        self._write(f"     Checkpoint: {ckpt_path}")
        self._write(f"     Time: {elapsed:.1f}s")
        self._write("")
    
    def log_training_complete(self, final_path):
        """Log at end of training completion."""
        self._write("="*70)
        self._write(f"✅ Training finished. Final model: {final_path}")
        self._write("="*70)
        self._write("")
    
    def log_pruning_info(self, pruning_rule):
        """Log the pruning rule being applied."""
        self._write(f"  Pruning pairs: {pruning_rule}")
    
    def clear_log(self):
        """Start fresh log (clear file)."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("")
