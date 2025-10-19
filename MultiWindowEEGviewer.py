"""
Multi-Window EEG Viewer - Temporal Analysis Tool
Compatible with dynamically-sized models from multi-window training
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from pathlib import Path
from collections import defaultdict
import random 

try:
    from datasets import load_dataset
except ImportError:
    print("Missing datasets library.")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEG_SAMPLE_RATE = 512

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
TARGET_IDS = set(TARGET_CATEGORIES.values())
ALL_COCO_IDS = list(range(1, 91)) 
EXCLUDED_IDS = set(ALL_COCO_IDS) - TARGET_IDS

# --- UPDATED ARCHITECTURE (DYNAMIC) ---

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
    """FIXED: Classifier built during init, not dynamically"""
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
        
        # Determine feature size with dummy forward
        with torch.no_grad():
            dummy_input = torch.randn(1, n_channels, n_timepoints)
            dummy_out = self._forward_features(dummy_input)
            feature_size = dummy_out.shape[1]
        
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

# --- DATA LOADER ---

class FilteredTestDataset:
    def __init__(self, annotations_path, max_samples=1000):
        print("Loading and filtering test dataset...")
        
        self.eeg_dataset = load_dataset("Alljoined/05_125", split='test', streaming=False)
        self.eeg_dataset = self.eeg_dataset.select(range(min(max_samples, len(self.eeg_dataset))))

        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        image_annotations = defaultdict(set)
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            image_annotations[img_id].add(ann['category_id'])

        self.filtered_samples = []
        for idx, sample in enumerate(self.eeg_dataset):
            coco_id = sample['coco_id']
            
            if coco_id in image_annotations:
                ann_ids = image_annotations[coco_id]
                contains_excluded = any(cat_id in EXCLUDED_IDS for cat_id in ann_ids)
                contains_target = any(cat_id in TARGET_IDS for cat_id in ann_ids)
                
                if not contains_excluded and contains_target:
                    self.filtered_samples.append({
                        'eeg_idx': idx,
                        'coco_id': coco_id,
                        'eeg_data': np.array(sample['EEG'], dtype=np.float32)
                    })
                    
        print(f"Loaded {len(self.filtered_samples)} filtered test samples.")
        if len(self.filtered_samples) == 0:
            raise RuntimeError("No suitable test samples found after filtering.")

    def get_eeg_window(self, sample_info, start_ms, end_ms):
        eeg_data = sample_info['eeg_data']
        # FIXED: Use float division for exact indices
        start_idx = int((start_ms / 1000.0) * EEG_SAMPLE_RATE)
        end_idx = int((end_ms / 1000.0) * EEG_SAMPLE_RATE)
        n_timepoints = end_idx - start_idx
        
        if eeg_data.shape[1] >= end_idx:
            eeg_window = eeg_data[:, start_idx:end_idx]
        else:
            eeg_window = eeg_data[:, start_idx:]
        
        # FIXED: Ensure exact length matches model (pad/truncate if needed)
        if eeg_window.shape[1] != n_timepoints:
            if eeg_window.shape[1] < n_timepoints:
                pad_width = n_timepoints - eeg_window.shape[1]
                eeg_window = np.pad(eeg_window, ((0,0), (0, pad_width)), mode='constant', constant_values=0)
            else:
                eeg_window = eeg_window[:, :n_timepoints]
        
        eeg_window = (eeg_window - eeg_window.mean(axis=1, keepdims=True)) / \
                     (eeg_window.std(axis=1, keepdims=True) + 1e-8)
        
        return eeg_window

    def get_random_sample_info(self):
        return random.choice(self.filtered_samples)

# --- VIEWER APPLICATION ---

class MultiWindowEEGViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Window EEG Viewer - Temporal Analysis")
        self.geometry("1600x900")
        
        self.models = {}
        self.coco_path = ""
        self.annotations_path = "" 
        self.test_data_filter = None 
        self.current_sample_info = None
        
        self.setup_gui()

    def setup_gui(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10, padx=10, fill=tk.X)
        
        ttk.Label(control_frame, text="COCO Path:").pack(side=tk.LEFT, padx=5)
        self.coco_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.coco_var, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Browse COCO", command=self.browse_coco).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Ann. Path:").pack(side=tk.LEFT, padx=5)
        self.ann_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.ann_var, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Browse Ann.", command=self.browse_ann).pack(side=tk.LEFT, padx=5)
        
        self.model_load_btn = ttk.Button(control_frame, text="Load Models from Folder", 
                                         command=self.load_models_from_folder)
        self.model_load_btn.pack(side=tk.LEFT, padx=20)
        
        self.test_btn = ttk.Button(control_frame, text="Test New Random Sample", 
                                   command=self.test_random_sample, 
                                   state=tk.DISABLED)
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(control_frame, text="Models: 0 loaded", fg="gray")
        self.status_label.pack(side=tk.LEFT, padx=20)

        main_content = ttk.Frame(self)
        main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        image_frame = ttk.Frame(main_content, width=400, height=400)
        image_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(image_frame, text="Current COCO Image", font=("Arial", 12, "bold")).pack(pady=5)
        self.image_canvas = tk.Canvas(image_frame, width=400, height=400, bg='lightgray')
        self.image_canvas.pack()
        self.coco_id_label = ttk.Label(image_frame, text="COCO ID: N/A")
        self.coco_id_label.pack(pady=5)
        self.pil_image_tk = None

        self.notebook = ttk.Notebook(main_content)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
    
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

    def load_models_from_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        
        if not self.annotations_path:
             messagebox.showerror("Error", "Please load the Annotations file first.")
             return

        self.models = {}
        model_files = sorted(Path(folder_path).glob('*.pth'))
        
        if not model_files:
            messagebox.showwarning("Warning", "No .pth model files found in the selected folder.")
            return

        for item in self.notebook.tabs():
            self.notebook.forget(item)

        total_loaded = 0
        try:
            for path in model_files:
                checkpoint = torch.load(path, map_location=DEVICE)
                
                start_ms = checkpoint.get('start_ms', 50)
                end_ms = checkpoint.get('end_ms', 350)
                n_layers = checkpoint.get('n_layers', 4)
                n_heads = checkpoint.get('n_heads', 8)
                time_label = checkpoint.get('label', f"{start_ms}-{end_ms}ms")
                
                # FIXED: Compute exact n_timepoints (matches trainer)
                start_idx = int((start_ms / 1000.0) * EEG_SAMPLE_RATE)
                end_idx = int((end_ms / 1000.0) * EEG_SAMPLE_RATE)
                n_timepoints = end_idx - start_idx
                print(f"Loading {time_label}: exact timepoints={n_timepoints}")
                
                model = HybridCNNTransformer(
                    n_timepoints=n_timepoints,
                    num_classes=len(TARGET_CATEGORIES),
                    n_layers=n_layers,
                    n_heads=n_heads
                ).to(DEVICE)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                model_key = f"{start_ms}-{end_ms}ms"
                self.models[model_key] = {
                    'model': model,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'label': time_label,
                    'auroc': checkpoint.get('val_auroc', 0.0)
                }
                
                self.create_model_tab(model_key)
                total_loaded += 1
            
            self.status_label.config(text=f"Models: {total_loaded} loaded", fg="green")
            self.test_data_filter = FilteredTestDataset(self.annotations_path)
            self.test_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{e}")
            import traceback
            print(traceback.format_exc())
            self.models = {}

    def create_model_tab(self, model_key):
        model_info = self.models[model_key]
        tab_title = f"{model_info['label']}"
        
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=tab_title)
        
        auroc_val = model_info['auroc']
        auroc_str = f"{auroc_val:.4f}" if isinstance(auroc_val, float) else "N/A"
        header_text = f"Window: {model_info['start_ms']}-{model_info['end_ms']}ms | AUROC: {auroc_str}"
        ttk.Label(tab_frame, text=header_text, font=("Arial", 12, "bold")).pack(pady=5)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        fig.suptitle(f"{model_info['label']}", fontsize=14)
        
        canvas = FigureCanvasTkAgg(fig, tab_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        model_info['fig'] = fig
        model_info['axes'] = axes
        model_info['canvas'] = canvas

    def _fetch_image(self, coco_id):
        formatted_id = f"{coco_id:012d}.jpg"
        paths = [os.path.join(self.coco_path, s, formatted_id) 
                for s in ["train2017", "val2017", "test2017"]]
        for path in paths:
            if os.path.exists(path):
                try:
                    return Image.open(path).convert("RGB")
                except:
                    pass
        return None

    def test_random_sample(self):
        if not self.models or self.test_data_filter is None:
            messagebox.showwarning("Setup Error", "Please load models first.")
            return

        try:
            self.current_sample_info = self.test_data_filter.get_random_sample_info()
            coco_id = self.current_sample_info['coco_id']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select sample: {e}")
            return

        image = self._fetch_image(coco_id)
        if image:
            self.display_image(image, coco_id)
        
        for model_key, info in self.models.items():
            self._process_single_model(model_key, info)
    
    def display_image(self, image, coco_id):
        w, h = image.size
        ratio = min(400/w, 400/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized_image = image.resize((new_w, new_h), Image.LANCZOS)
        
        self.pil_image_tk = ImageTk.PhotoImage(resized_image)
        self.image_canvas.create_image(200, 200, image=self.pil_image_tk, anchor=tk.CENTER)
        self.coco_id_label.config(text=f"COCO ID: {coco_id}")
        
    def _process_single_model(self, model_key, info):
        eeg_window_np = self.test_data_filter.get_eeg_window(
            self.current_sample_info,
            info['start_ms'],
            info['end_ms']
        )
        eeg_tensor = torch.from_numpy(eeg_window_np).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            logits = info['model'](eeg_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:20]
        cat_list = list(TARGET_CATEGORIES.keys())
        top_20_categories = [(cat_list[i], probs[i]) for i in top_indices]

        self.update_tab_visualization(info, eeg_window_np, top_20_categories, probs)

    def update_tab_visualization(self, info, eeg_data, top_categories, all_probs):
        fig = info['fig']
        ax_heatmap, ax_bar = info['axes']
        
        ax_heatmap.clear()
        ax_heatmap.imshow(eeg_data, aspect='auto', cmap='RdBu_r', 
                           interpolation='nearest', vmin=-3, vmax=3) 
        
        ax_heatmap.set_title(f"EEG ({info['start_ms']}-{info['end_ms']}ms)", fontsize=10)
        ax_heatmap.set_xlabel("Time (ms)")
        ax_heatmap.set_ylabel("Channel")

        ax_bar.clear()
        categories, confidences = zip(*top_categories)
        
        ax_bar.barh(range(len(categories)), confidences, color='steelblue')
        ax_bar.set_yticks(range(len(categories)))
        ax_bar.set_yticklabels(categories, fontsize=8)
        ax_bar.set_xlabel("Probability")
        ax_bar.set_title("Top 20 Predictions", fontsize=10)
        ax_bar.set_xlim(0, 1)
        ax_bar.invert_yaxis()

        fig.tight_layout()
        info['canvas'].draw()

if __name__ == "__main__":
    app = MultiWindowEEGViewer()
    app.mainloop()