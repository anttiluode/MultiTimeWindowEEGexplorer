"""
Enhanced EEG Weak Signal Detector - FAST VERSION
Applies zero-copy optimizations to the proven architecture
NEW: Supports training of multiple time-window models sequentially.
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import threading
import queue
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch.amp import autocast, GradScaler

try:
    from datasets import load_dataset
    torch.backends.cudnn.benchmark = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install datasets scikit-learn")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEG_SAMPLE_RATE = 512

# SPEED SETTINGS
FAST_BATCH_SIZE = 256
FAST_NUM_WORKERS = 0

# --- NEW: FIXED TIME WINDOWS FOR TEMPORAL ANALYSIS ---
# Format: (start_ms, end_ms, label)
FIXED_TIME_WINDOWS = [
    (50, 150, "EarlyVisual"),    # P100/N100 range
    (150, 250, "MidFeature"),     # N170/P200 range
    (250, 350, "LateSemantic"),   # N400/LPC range (Your successful window)
    (50, 250, "EarlyCombined"),   # For comparison
    (50, 350, "FullWindow")       # Baseline
]

TARGET_CATEGORIES = {
    'elephant': 22, 'giraffe': 25, 'bear': 23, 'zebra': 24,
    'cow': 21, 'sheep': 20, 'horse': 19, 'dog': 18, 'cat': 17, 'bird': 16,
    'airplane': 5, 'train': 7, 'boat': 9, 'bus': 6, 'truck': 8,
    'motorcycle': 4, 'bicycle': 2, 'car': 3,
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15,
    'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
    'kite': 38, 'baseball bat': 39, 'skateboard': 41, 'surfboard': 42,
    'banana': 52, 'apple': 53, 'orange': 55, 'broccoli': 56,
    'pizza': 59, 'donut': 60, 'cake': 61,
}

CATEGORY_NAMES = {v: k for k, v in TARGET_CATEGORIES.items()}

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        residual = x
        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        output = self.dropout(output)
        return self.layer_norm(residual + output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(residual + x)


class HybridCNNTransformer(nn.Module):
    def __init__(self, n_channels=64, n_timepoints=154, num_classes=len(TARGET_CATEGORIES),
                 d_model=256, n_heads=8, n_layers=4, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(256, d_model, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(d_model)
        temp_size = n_timepoints
        for _ in range(2):
            temp_size = temp_size // 2
        self.seq_len = int(temp_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadSelfAttention(d_model, n_heads, dropout),
                'feedforward': FeedForward(d_model, d_model * 4, dropout)
            }) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * self.seq_len, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        x = self.pool2(F.elu(self.bn2(self.conv2(x))))
        x = F.elu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)
        x = x + self.pos_encoding
        for layer in self.transformer_layers:
            x = layer['attention'](x)
            x = layer['feedforward'](x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


class FastEEGDataset(Dataset):
    """OPTIMIZED: Pre-cache as tensors, not numpy"""
    def __init__(self, coco_path, annotations_path, split='train', max_samples=None,
                 start_ms=50, end_ms=350, trials_to_average=1):
        self.coco_path = Path(coco_path)
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.trials_to_average = trials_to_average
        
        print(f"Loading Alljoined ({split}) with window {start_ms}-{end_ms}ms...")
        self.dataset = load_dataset("Alljoined/05_125", split=split, streaming=False)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(int(max_samples), len(self.dataset))))
        
        print(f"Loading COCO annotations...")
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        self.image_categories = defaultdict(set)
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            if cat_id in CATEGORY_NAMES:
                self.image_categories[img_id].add(cat_id)
        
        # OPTIMIZED: Pre-cache as TENSORS
        print("Pre-caching EEG data as tensors...")
        self.cached_eeg = {}
        self.category_samples = defaultdict(list)
        
        for idx, sample in enumerate(self.dataset):
            coco_id = sample['coco_id']
            if coco_id in self.image_categories and len(self.image_categories[coco_id]) > 0:
                label = torch.zeros(len(TARGET_CATEGORIES))
                for cat_id in self.image_categories[coco_id]:
                    if cat_id in CATEGORY_NAMES:
                        cat_idx = list(TARGET_CATEGORIES.values()).index(cat_id)
                        label[cat_idx] = 1.0
                
                if label.sum() > 0:
                    # Cache as tensor immediately
                    eeg_tensor = self._get_eeg_tensor(sample)
                    self.cached_eeg[idx] = eeg_tensor
                    
                    for cat_id in self.image_categories[coco_id]:
                        if cat_id in CATEGORY_NAMES:
                            cat_idx = list(TARGET_CATEGORIES.values()).index(cat_id)
                            self.category_samples[cat_idx].append((idx, label))
        
        self.samples = []
        for cat_idx, samples in self.category_samples.items():
            self.samples.extend(samples)
        
        print(f"Cached {len(self.cached_eeg)} EEG tensors, {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_eeg_tensor(self, sample):
        """Extract and convert to tensor in one pass"""
        eeg_data = np.array(sample['EEG'], dtype=np.float32)
        start_idx = int((self.start_ms / 1000) * EEG_SAMPLE_RATE)
        end_idx = int((self.end_ms / 1000) * EEG_SAMPLE_RATE)
        
        if eeg_data.shape[1] >= end_idx:
            eeg_window = eeg_data[:, start_idx:end_idx]
        else:
            eeg_window = eeg_data[:, start_idx:]
        
        # Z-score
        eeg_window = (eeg_window - eeg_window.mean(axis=1, keepdims=True)) / \
                     (eeg_window.std(axis=1, keepdims=True) + 1e-8)
        
        return torch.from_numpy(eeg_window).float()
    
    def __getitem__(self, idx):
        sample_idx, label = self.samples[idx]
        
        if self.trials_to_average > 1:
            present_cats = (label == 1).nonzero(as_tuple=True)[0]
            cat_idx = present_cats[torch.randint(len(present_cats), (1,))].item()
            available_samples = self.category_samples[cat_idx]
            
            n_to_sample = min(self.trials_to_average, len(available_samples))
            replace_needed = n_to_sample > len(available_samples)
            selected = np.random.choice(len(available_samples), size=n_to_sample, replace=replace_needed)
            
            # Stack and average tensors (faster than numpy)
            eeg_tensors = [self.cached_eeg[available_samples[s][0]] for s in selected]
            eeg = torch.stack(eeg_tensors).mean(dim=0)
        else:
            eeg = self.cached_eeg[sample_idx].clone()
        
        # In-place augmentation
        if np.random.rand() > 0.5:
            eeg.add_(torch.randn_like(eeg) * 0.05)
        
        return eeg, label


class EnhancedTrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Enhanced EEG Signal Trainer - FAST")
        self.geometry("1200x950")
        
        self.coco_path = ""
        self.annotations_path = ""
        self.model = None
        self.train_thread = None
        self.stop_flag = threading.Event()
        self.log_queue = queue.Queue()
        self.scaler = GradScaler('cuda')
        
        self.setup_gui()
        self.process_logs()
    
    def setup_gui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        train_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text="Training")
        self.setup_train_tab(train_tab)
        
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Statistical Analysis")
        self.setup_analysis_tab(analysis_tab)
    
    def setup_train_tab(self, parent):
        title = tk.Label(parent, text="Hybrid CNN-Transformer (FAST)", 
                        font=("Arial", 14, "bold"))
        title.pack(pady=5)
        
        info = tk.Label(parent, 
                       text=f"Speed optimizations: Tensor cache + AMP + Batch={FAST_BATCH_SIZE}\n"
                            "Proven architecture: 4 layers (AUROC 0.96)",
                       fg="green", font=("Arial", 9))
        info.pack(pady=5)
        
        # Paths
        path_frame = ttk.LabelFrame(parent, text="Dataset")
        path_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(path_frame, text="COCO:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.coco_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.coco_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_coco).grid(row=0, column=2)
        
        tk.Label(path_frame, text="Annotations:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.ann_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.ann_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_ann).grid(row=1, column=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(parent, text="Configuration")
        settings_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(settings_frame, text="Max Samples:").grid(row=0, column=0, padx=5)
        self.max_var = tk.IntVar(value=4000)
        tk.Spinbox(settings_frame, from_=1000, to=10000, increment=1000,
                  textvariable=self.max_var, width=10).grid(row=0, column=1)
        
        tk.Label(settings_frame, text="Epochs:").grid(row=0, column=2, padx=5)
        self.epochs_var = tk.IntVar(value=1000) # Increased default for multi-window
        tk.Spinbox(settings_frame, from_=200, to=2000, increment=100,
                  textvariable=self.epochs_var, width=10).grid(row=0, column=3)
        
        tk.Label(settings_frame, text="Transformer Layers:").grid(row=1, column=0, padx=5)
        self.n_layers_var = tk.IntVar(value=4)
        tk.Spinbox(settings_frame, from_=2, to=8, increment=1,
                  textvariable=self.n_layers_var, width=10).grid(row=1, column=1)
        
        tk.Label(settings_frame, text="Attention Heads:").grid(row=1, column=2, padx=5)
        self.n_heads_var = tk.IntVar(value=8)
        tk.Spinbox(settings_frame, from_=4, to=16, increment=4,
                  textvariable=self.n_heads_var, width=10).grid(row=1, column=3)
        
        tk.Label(settings_frame, text="Trials to Average:").grid(row=2, column=0, padx=5)
        self.avg_trials_var = tk.IntVar(value=2)
        tk.Spinbox(settings_frame, from_=1, to=5, increment=1,
                  textvariable=self.avg_trials_var, width=10).grid(row=2, column=1)
        
        # --- REMOVED Single Window controls (row 2, columns 2 & 3) ---
        tk.Label(settings_frame, text="Window: (Multi-Mode)").grid(row=2, column=2, columnspan=2, padx=5)
        
        # Buttons
        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=5)
        
        # --- NEW: Multi-Train Button ---
        self.train_btn = tk.Button(btn_frame, text="Train 5 Multi-Window Models", 
                                   command=self.start_multi_train,
                                   bg="#9C27B0", fg="white", font=("Arial", 10, "bold"))
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop", 
                                  command=self.stop_train,
                                  bg="#f44336", fg="white",
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = tk.Button(btn_frame, text="Statistical Analysis (WIP)",
                                     command=self.analyze_signals,
                                     bg="#2196F3", fg="white", state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(parent, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(parent, text="Training Log")
        log_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=12, bg='black', fg='lightgreen',
                               font=('Courier', 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    # ... (setup_analysis_tab, browse_coco, browse_ann, log, process_logs remain the same)
    def setup_analysis_tab(self, parent):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def browse_coco(self):
        path = filedialog.askdirectory()
        if path:
            self.coco_var.set(path)
            self.coco_path = path
    
    def browse_ann(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.ann_var.set(path)
            self.annotations_path = path
    
    def log(self, msg):
        self.log_queue.put(msg)
    
    def process_logs(self):
        try:
            while not self.log_queue.empty():
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self.process_logs)

    # --- NEW: Multi-Train Entry Point ---
    def start_multi_train(self):
        if not self.coco_path or not self.annotations_path:
            messagebox.showerror("Error", "Select paths first")
            return
        
        self.stop_flag.clear()
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.train_thread = threading.Thread(target=self._multi_train_loop, daemon=True)
        self.train_thread.start()
        
    def stop_train(self):
        self.stop_flag.set()

    # --- NEW: Master Loop to Train All Windows ---
    def _multi_train_loop(self):
        try:
            n_models = len(FIXED_TIME_WINDOWS)
            total_epochs = self.epochs_var.get()
            
            for i, (start_ms, end_ms, label) in enumerate(FIXED_TIME_WINDOWS):
                if self.stop_flag.is_set():
                    self.log("\nMulti-training interrupted by user.")
                    break
                
                # Calculate progress range for the current model
                start_prog = (i / n_models) * 100
                end_prog = ((i + 1) / n_models) * 100

                self.log(f"\n{'='*70}")
                self.log(f"STARTING MODEL {i+1}/{n_models}: {label} ({start_ms}-{end_ms}ms)")
                self.log(f"{'='*70}")

                # Call the core training routine for this single window
                self._single_train_loop(
                    start_ms, end_ms, label, total_epochs, start_prog, end_prog
                )
            
            self.log("\n" + "="*70)
            self.log("ALL MULTI-WINDOW MODELS TRAINED.")
            self.log("="*70)
            
        except Exception as e:
            self.log(f"MASTER ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            
        finally:
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            # Re-enable analyze button if models were trained (WIP status set in old analyze_signals)

    # --- REVISED: Single Training Loop to be called by Master Loop ---
    def _single_train_loop(self, start_ms, end_ms, label, total_epochs, start_prog, end_prog):
        
        n_layers = self.n_layers_var.get()
        n_heads = self.n_heads_var.get()
        trials_avg = self.avg_trials_var.get()
        
        n_timepoints = int(((end_ms - start_ms) / 1000) * EEG_SAMPLE_RATE)
        
        # 1. Setup Model (New instance for each window)
        model = HybridCNNTransformer(
            n_timepoints=n_timepoints,
            num_classes=len(TARGET_CATEGORIES),
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=0.3
        ).to(DEVICE)
        
        params = sum(p.numel() for p in model.parameters())
        self.log(f"  Parameters: {params:,} | Timepoints: {n_timepoints}")
        
        # 2. Setup Data (New Dataset instance for each window)
        dataset = FastEEGDataset(
            self.coco_path,
            self.annotations_path,
            'train',
            int(self.max_var.get() * 1.25),
            start_ms=start_ms,
            end_ms=end_ms,
            trials_to_average=trials_avg
        )
        total = len(dataset)
        train_size = int(0.8 * total)
        val_size = total - train_size
        
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.log(f"  Train: {train_size}, Val: {val_size}")
        
        train_loader = DataLoader(train_set, batch_size=FAST_BATCH_SIZE, shuffle=True, 
                                 num_workers=FAST_NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=FAST_BATCH_SIZE, shuffle=False, 
                               num_workers=FAST_NUM_WORKERS, pin_memory=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2
        )
        
        best_val_auroc = 0.0
        best_val_loss = float('inf')

        # 3. Training Loop
        for epoch in range(total_epochs):
            if self.stop_flag.is_set():
                break
            
            # Training
            model.train()
            train_loss = 0
            for eeg, labels in train_loader:
                if self.stop_flag.is_set(): break
                eeg = eeg.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                with autocast('cuda'):
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            all_val_probs = []
            all_val_labels = []
            with torch.no_grad():
                for eeg, labels in val_loader:
                    eeg = eeg.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    with autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    val_loss += loss.item()
                    probs = torch.sigmoid(logits.float())
                    all_val_probs.append(probs.cpu().numpy())
                    all_val_labels.append(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            val_probs = np.concatenate(all_val_probs)
            val_labels = np.concatenate(all_val_labels)
            try:
                val_auroc = roc_auc_score(val_labels, val_probs, average='macro')
                auroc_str = f"AUROC={val_auroc:.4f}"
            except ValueError:
                val_auroc = 0.0
                auroc_str = "AUROC=N/A"
            
            scheduler.step()
            
            # Update overall progress
            current_model_progress = (epoch + 1) / total_epochs
            self.progress['value'] = start_prog + current_model_progress * (end_prog - start_prog)
            
            self.log(f"  Epoch {epoch+1}/{total_epochs}: "
                    f"TrLoss={train_loss:.4f} ValLoss={val_loss:.4f} {auroc_str} ")
            
            is_better = (val_auroc > best_val_auroc) if val_auroc > 0 else (val_loss < best_val_loss)

            if is_better:
                best_val_loss = val_loss
                best_val_auroc = val_auroc
                
                # --- NEW: Unique Filename for each window ---
                model_filename = f"signal_detector_{start_ms}_{end_ms}ms_{label}.pth"
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_auroc': val_auroc,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'time_label': label,
                }, model_filename)
                self.log(f"  -> SAVED BEST: {model_filename} (AUROC={val_auroc:.4f})")
        
        self.log(f"  COMPLETE: Best AUROC for {label} was {best_val_auroc:.4f}")


    # --- analyze_signals kept for reference, but disabled in GUI for multi-train mode ---
    def analyze_signals(self):
        # This function would need to be re-written to load ALL trained models
        # for a comprehensive analysis, which is complex and outside this request's scope.
        # It is kept here as a disabled placeholder.
        messagebox.showinfo("WIP", "Statistical Analysis must be adapted for multi-model output. Please analyze the individual .pth files.")
        pass

if __name__ == "__main__":
    app = EnhancedTrainerGUI()
    app.mainloop()