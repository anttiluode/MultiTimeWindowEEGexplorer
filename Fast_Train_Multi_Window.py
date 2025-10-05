"""
Multi-Window EEG Trainer - Sequential Training Across Time Windows
Trains separate models for different ERP time ranges to analyze temporal dynamics
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
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEG_SAMPLE_RATE = 512
FAST_BATCH_SIZE = 256
FAST_NUM_WORKERS = 0

# Time windows for temporal analysis
TIME_WINDOWS = [
    (50, 150, "EarlyVisual"),     # P100/N100
    (150, 250, "MidFeature"),     # N170/P200
    (250, 350, "LateSemantic"),   # N400/P300
    (50, 250, "EarlyCombined"),   # Early+Mid
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
        
        self.d_model = d_model
        self.dropout_val = dropout
        
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadSelfAttention(d_model, n_heads, dropout),
                'feedforward': FeedForward(d_model, d_model * 4, dropout)
            }) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # FIXED: Determine actual feature size with dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, n_channels, n_timepoints)
            dummy_out = self._forward_features(dummy_input)
            feature_size = dummy_out.shape[1]
        
        # Build classifier with known feature size
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def _forward_features(self, x):
        """Feature extraction without classifier"""
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        x = self.pool2(F.elu(self.bn2(self.conv2(x))))
        x = F.elu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)
        
        batch_size, seq_len, d_model = x.shape
        
        if not hasattr(self, 'pos_encoding') or self.pos_encoding.shape[1] != seq_len:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, seq_len, d_model, device=x.device) * 0.02
            )
        
        x = x + self.pos_encoding
        
        for layer in self.transformer_layers:
            x = layer['attention'](x)
            x = layer['feedforward'](x)
        
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        return x
        
    def forward(self, x):
        x = self._forward_features(x)
        return self.classifier(x)

class FastEEGDataset(Dataset):
    """Simple dataset - no synthetic data generation"""
    def __init__(self, coco_path, annotations_path, split='train', max_samples=None,
                 start_ms=50, end_ms=350, trials_to_average=1):
        self.coco_path = Path(coco_path)
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.trials_to_average = trials_to_average
        
        # FIXED: Compute exact n_timepoints for consistency
        self.start_idx = int((start_ms / 1000.0) * EEG_SAMPLE_RATE)
        self.end_idx = int((end_ms / 1000.0) * EEG_SAMPLE_RATE)
        self.n_timepoints = self.end_idx - self.start_idx
        
        print(f"Loading Alljoined ({split}) with window {start_ms}-{end_ms}ms (exact length: {self.n_timepoints})...")
        self.dataset = load_dataset("Alljoined/05_125", split=split, streaming=False)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(int(max_samples), len(self.dataset))))
        
        print(f"Loading COCO annotations...")
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        self.image_categories = defaultdict(set)
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if ann['category_id'] in CATEGORY_NAMES:
                self.image_categories[img_id].add(ann['category_id'])
        
        # Pre-cache as tensors
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
        eeg_data = np.array(sample['EEG'], dtype=np.float32)
        
        if eeg_data.shape[1] >= self.end_idx:
            eeg_window = eeg_data[:, self.start_idx:self.end_idx]
        else:
            eeg_window = eeg_data[:, self.start_idx:]
        
        # Ensure exact length matches model expectation
        if eeg_window.shape[1] != self.n_timepoints:
            # Pad or truncate if dataset edge case (rare, but safe)
            if eeg_window.shape[1] < self.n_timepoints:
                pad_width = self.n_timepoints - eeg_window.shape[1]
                eeg_window = np.pad(eeg_window, ((0,0), (0, pad_width)), mode='constant', constant_values=0)
            else:
                eeg_window = eeg_window[:, :self.n_timepoints]
        
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
            selected = np.random.choice(len(available_samples), size=n_to_sample, replace=True)
            
            eeg_tensors = [self.cached_eeg[available_samples[s][0]] for s in selected]
            eeg = torch.stack(eeg_tensors).mean(dim=0)
        else:
            eeg = self.cached_eeg[sample_idx].clone()
        
        if np.random.rand() > 0.5:
            eeg.add_(torch.randn_like(eeg) * 0.05)
        
        return eeg, label

class MultiWindowTrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Window EEG Trainer")
        self.geometry("1200x950")
        
        self.coco_path = ""
        self.annotations_path = ""
        self.train_thread = None
        self.stop_flag = threading.Event()
        self.log_queue = queue.Queue()
        
        self.setup_gui()
        self.process_logs()
    
    def setup_gui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        train_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text="Training")
        self.setup_train_tab(train_tab)
    
    def setup_train_tab(self, parent):
        title = tk.Label(parent, text="Multi-Window Temporal Analysis", 
                        font=("Arial", 14, "bold"))
        title.pack(pady=5)
        
        info = tk.Label(parent, 
                       text="Trains 5 separate models for different ERP time ranges",
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
        settings_frame = ttk.LabelFrame(parent, text="Training Settings")
        settings_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(settings_frame, text="Max Samples:").grid(row=0, column=0, padx=5)
        self.max_var = tk.IntVar(value=4000)
        tk.Spinbox(settings_frame, from_=1000, to=10000, increment=1000,
                  textvariable=self.max_var, width=10).grid(row=0, column=1)
        
        tk.Label(settings_frame, text="Epochs per model:").grid(row=0, column=2, padx=5)
        self.epochs_var = tk.IntVar(value=1000)
        tk.Spinbox(settings_frame, from_=100, to=2000, increment=100,
                  textvariable=self.epochs_var, width=10).grid(row=0, column=3)
        
        tk.Label(settings_frame, text="Trials to Average:").grid(row=1, column=0, padx=5)
        self.avg_trials_var = tk.IntVar(value=2)
        tk.Spinbox(settings_frame, from_=1, to=5, increment=1,
                  textvariable=self.avg_trials_var, width=10).grid(row=1, column=1)
        
        # Time windows display
        windows_frame = ttk.LabelFrame(parent, text="Time Windows")
        windows_frame.pack(pady=5, padx=10, fill=tk.X)
        
        for start, end, label in TIME_WINDOWS:
            tk.Label(windows_frame, text=f"{label}: {start}-{end}ms", 
                    font=("Courier", 9)).pack(anchor=tk.W, padx=10)
        
        # Buttons
        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=5)
        
        self.train_btn = tk.Button(btn_frame, text="Train All 5 Models", 
                                   command=self.start_train,
                                   bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop", 
                                  command=self.stop_train,
                                  bg="#f44336", fg="white",
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(parent, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(parent, text="Training Log")
        log_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=15, bg='black', fg='lightgreen',
                               font=('Courier', 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
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
    
    def start_train(self):
        if not self.coco_path or not self.annotations_path:
            messagebox.showerror("Error", "Select paths first")
            return
        
        self.stop_flag.clear()
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.train_thread = threading.Thread(target=self._train_all_windows, daemon=True)
        self.train_thread.start()
    
    def stop_train(self):
        self.stop_flag.set()
    
    def _train_all_windows(self):
        try:
            self.log("="*70)
            self.log("MULTI-WINDOW TEMPORAL ANALYSIS")
            self.log("="*70)
            
            n_windows = len(TIME_WINDOWS)
            
            for i, (start_ms, end_ms, label) in enumerate(TIME_WINDOWS):
                if self.stop_flag.is_set():
                    break
                
                self.log(f"\n{'='*70}")
                self.log(f"MODEL {i+1}/{n_windows}: {label} ({start_ms}-{end_ms}ms)")
                self.log(f"{'='*70}")
                
                self._train_single_window(start_ms, end_ms, label, i, n_windows)
            
            self.log("\n" + "="*70)
            self.log("ALL MODELS COMPLETE")
            self.log("="*70)
            
        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def _train_single_window(self, start_ms, end_ms, label, window_idx, total_windows):
        # FIXED: Compute exact n_timepoints from idx difference
        start_idx = int((start_ms / 1000.0) * EEG_SAMPLE_RATE)
        end_idx = int((end_ms / 1000.0) * EEG_SAMPLE_RATE)
        n_timepoints = end_idx - start_idx
        
        # Create model
        model = HybridCNNTransformer(
            n_timepoints=n_timepoints,
            num_classes=len(TARGET_CATEGORIES),
            n_layers=4,
            n_heads=8,
            dropout=0.3
        ).to(DEVICE)
        
        self.log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.log(f"Exact timepoints: {n_timepoints}")
        
        # Create dataset
        dataset = FastEEGDataset(
            self.coco_path,
            self.annotations_path,
            'train',
            int(self.max_var.get() * 1.25),
            start_ms=start_ms,
            end_ms=end_ms,
            trials_to_average=self.avg_trials_var.get()
        )
        
        total = len(dataset)
        train_size = int(0.8 * total)
        val_size = total - train_size
        
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.log(f"Train: {train_size}, Val: {val_size}")
        
        train_loader = DataLoader(train_set, batch_size=FAST_BATCH_SIZE, shuffle=True, 
                                 num_workers=FAST_NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=FAST_BATCH_SIZE, shuffle=False, 
                               num_workers=FAST_NUM_WORKERS, pin_memory=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        scaler = GradScaler('cuda')
        
        best_val_auroc = 0.0
        
        for epoch in range(self.epochs_var.get()):
            if self.stop_flag.is_set():
                break
            
            # Train
            model.train()
            train_loss = 0
            for eeg, labels in train_loader:
                if self.stop_flag.is_set():
                    break
                eeg = eeg.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                with autocast('cuda'):
                    logits = model(eeg)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_loss = 0
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for eeg, labels in val_loader:
                    eeg = eeg.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    with autocast('cuda'):
                        logits = model(eeg)
                        loss = criterion(logits, labels)
                    val_loss += loss.item()
                    probs = torch.sigmoid(logits.float())
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            try:
                val_auroc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_probs), average='macro')
            except:
                val_auroc = 0.0
            
            scheduler.step()
            
            # Progress
            model_progress = (epoch + 1) / self.epochs_var.get()
            total_progress = (window_idx + model_progress) / total_windows * 100
            self.progress['value'] = total_progress
            
            if epoch % 10 == 0:
                self.log(f"Epoch {epoch+1}/{self.epochs_var.get()}: "
                        f"TrLoss={train_loss:.4f} ValLoss={val_loss:.4f} AUROC={val_auroc:.4f}")
            
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                filename = f"model_{start_ms}_{end_ms}ms_{label}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_auroc': val_auroc,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'label': label
                }, filename)
                if epoch % 10 == 0:
                    self.log(f"  -> Saved: {filename}")
        
        self.log(f"Best AUROC: {best_val_auroc:.4f}")

if __name__ == "__main__":
    app = MultiWindowTrainerGUI()
    app.mainloop()
