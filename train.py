"""
Training functions for Dual-Region Network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, classification_report

from models import DualRegionNet
from data import normalize_data_robust, enhance_features_advanced


class DualInputDataset(Dataset):
    """Dataset for dual-region inputs."""
    def __init__(self, ob_data, pcx_data, labels):
        self.ob_data = ob_data
        self.pcx_data = pcx_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ob_data[idx], self.pcx_data[idx], self.labels[idx]


def mixup_data(ob_x, pcx_x, y, alpha=0.2, device=None):
    """Apply Mixup augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = ob_x.size(0)
    index = torch.randperm(batch_size).to(device or ob_x.device)
    mixed_ob = lam * ob_x + (1 - lam) * ob_x[index]
    mixed_pcx = lam * pcx_x + (1 - lam) * pcx_x[index]
    return mixed_ob, mixed_pcx, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model(model, train_loader, val_loader, device, config, fold=1):
    """Train a single model for one fold."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = OneCycleLR(optimizer, max_lr=config['learning_rate'] * 10,
                           epochs=config['num_epochs'], steps_per_epoch=len(train_loader),
                           pct_start=0.1, anneal_strategy='cos')

    best_val_acc = 0.0
    best_model_state = None
    mixup_alpha = config.get('mixup_alpha', 0.2)

    for epoch in range(config['num_epochs']):
        model.train()
        train_correct = train_total = 0

        for ob_batch, pcx_batch, labels in train_loader:
            ob_batch = ob_batch.to(device, non_blocking=True)
            pcx_batch = pcx_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if config.get('use_mixup', False):
                ob_mixed, pcx_mixed, y_a, y_b, lam = mixup_data(
                    ob_batch, pcx_batch, labels, mixup_alpha, device)
                outputs = model(ob_mixed, pcx_mixed)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(ob_batch, pcx_batch)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for ob_batch, pcx_batch, labels in val_loader:
                ob_batch = ob_batch.to(device, non_blocking=True)
                pcx_batch = pcx_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(ob_batch, pcx_batch)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Fold {fold} Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Train={train_acc:.2f}%, Val={val_acc:.2f}%")

    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    return model, best_val_acc, []


def train_ensemble(X_ob, X_pcx, y_encoded, config, label_encoder, device=None, model_kwargs=None):
    """Train an ensemble with K-fold CV."""
    print("\n=== Training Ensemble ===")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_kwargs is None:
        model_kwargs = {}

    num_folds = config['num_folds']
    num_classes = len(np.unique(y_encoded))

    class_indices = [np.where(y_encoded == i)[0] for i in range(num_classes)]
    min_class_size = min(len(idx) for idx in class_indices)

    test_size_per_class = 50
    train_size_per_class = min_class_size - test_size_per_class

    print(f"Balanced splits: {train_size_per_class} train, {test_size_per_class} test per class")
    if model_kwargs:
        print(f"Ablation config: {model_kwargs}")

    # Create test split
    np.random.seed(42)
    test_indices, remaining_indices = [], []
    for cls in range(num_classes):
        idx = class_indices[cls].copy()
        np.random.shuffle(idx)
        test_indices.extend(idx[:test_size_per_class])
        remaining_indices.extend(idx[test_size_per_class:])

    X_ob_test = X_ob[test_indices]
    X_pcx_test = X_pcx[test_indices]
    y_test = y_encoded[test_indices]

    class_remaining = [[] for _ in range(num_classes)]
    for idx in remaining_indices:
        class_remaining[y_encoded[idx]].append(idx)

    samples_per_fold = [len(idx) // num_folds for idx in class_remaining]

    models, val_accs, test_logits = [], [], []

    for fold_idx in range(num_folds):
        print(f"\n=== Fold {fold_idx+1}/{num_folds} ===")
        
        train_indices, val_indices = [], []
        for cls in range(num_classes):
            cls_idx = class_remaining[cls]
            fold_size = samples_per_fold[cls]
            start = fold_idx * fold_size
            end = (fold_idx + 1) * fold_size if fold_idx < num_folds - 1 else len(cls_idx)
            val_indices.extend(cls_idx[start:end])
            train_indices.extend(cls_idx[:start] + cls_idx[end:])

        X_ob_tr, X_pcx_tr, y_tr = X_ob[train_indices], X_pcx[train_indices], y_encoded[train_indices]
        X_ob_va, X_pcx_va, y_va = X_ob[val_indices], X_pcx[val_indices], y_encoded[val_indices]

        X_ob_tr_n, X_ob_va_n, _ = normalize_data_robust(X_ob_tr, X_ob_va)
        X_pcx_tr_n, X_pcx_va_n, _ = normalize_data_robust(X_pcx_tr, X_pcx_va)
        _, _, X_ob_te_n = normalize_data_robust(X_ob_tr, None, X_ob_test)
        _, _, X_pcx_te_n = normalize_data_robust(X_pcx_tr, None, X_pcx_test)

        X_ob_tr_e = enhance_features_advanced(X_ob_tr_n)
        X_ob_va_e = enhance_features_advanced(X_ob_va_n)
        X_ob_te_e = enhance_features_advanced(X_ob_te_n)
        X_pcx_tr_e = enhance_features_advanced(X_pcx_tr_n)
        X_pcx_va_e = enhance_features_advanced(X_pcx_va_n)
        X_pcx_te_e = enhance_features_advanced(X_pcx_te_n)

        tr_ds = DualInputDataset(torch.FloatTensor(X_ob_tr_e), torch.FloatTensor(X_pcx_tr_e), torch.LongTensor(y_tr))
        va_ds = DualInputDataset(torch.FloatTensor(X_ob_va_e), torch.FloatTensor(X_pcx_va_e), torch.LongTensor(y_va))
        te_ds = DualInputDataset(torch.FloatTensor(X_ob_te_e), torch.FloatTensor(X_pcx_te_e), torch.LongTensor(y_test))

        train_loader = DataLoader(tr_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(va_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(te_ds, batch_size=config['batch_size'], shuffle=False)

        model = DualRegionNet(
            X_ob_tr_e.shape[1], X_ob_tr_e.shape[2],
            X_pcx_tr_e.shape[1], X_pcx_tr_e.shape[2],
            num_classes, dropout_rate=config['dropout_rate'],
            **model_kwargs
        ).to(device)

        model, val_acc, _ = train_model(model, train_loader, val_loader, device, config, fold=fold_idx+1)
        models.append(model)
        val_accs.append(val_acc)

        model.eval()
        fold_logits = []
        with torch.no_grad():
            for ob_b, pcx_b, _ in test_loader:
                logits = model(ob_b.to(device), pcx_b.to(device))
                fold_logits.append(logits.cpu().numpy())
        test_logits.append(np.vstack(fold_logits))

    # Ensemble
    ensemble_logits = np.mean(test_logits, axis=0)
    ensemble_logits -= ensemble_logits.max(axis=1, keepdims=True)
    ensemble_probs = np.exp(ensemble_logits)
    ensemble_probs /= ensemble_probs.sum(axis=1, keepdims=True)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    
    print(f"\n=== Results ===")
    print(f"Val Accs: {val_accs}")
    print(f"Mean Val: {np.mean(val_accs):.2f}%")
    print(f"Test Acc: {ensemble_acc * 100:.2f}%")

    return models, ensemble_acc, {
        'val_accs': val_accs,
        'val_mean': np.mean(val_accs),
        'val_std': np.std(val_accs, ddof=1) if len(val_accs) > 1 else 0.0,
        'val_sem': np.std(val_accs) / np.sqrt(len(val_accs)),
        'test_acc': ensemble_acc,
    }