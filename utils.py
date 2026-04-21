import torch
import torch.nn as nn
import numpy as np
import random
import os
import yaml
import time
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buf_size   = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buf_size) / 1024 ** 2


def safe_mape(y_true, y_pred) -> float:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    yt = y_true.flatten()
    yp = y_pred.flatten()
    mask = np.isfinite(yt) & np.isfinite(yp) & (np.abs(yt) > 0.01)
    if mask.sum() == 0:
        return 100.0
    return float(np.clip(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100, 0.0, 200.0))


def denormalize_data(data, mean, std):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return data * std + mean


def normalize_data(data, mean, std):
    return (data - mean) / std


class MetricsCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_predictions: List[float] = []
        self.all_targets: List[float] = []
        self.losses: List[float] = []

    def update(self, pred, target, loss: float = None):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        pf, tf = pred.flatten(), target.flatten()
        mask = np.isfinite(pf) & np.isfinite(tf)
        if mask.sum() > 0:
            self.all_predictions.extend(pf[mask].tolist())
            self.all_targets.extend(tf[mask].tolist())
        if loss is not None and np.isfinite(loss):
            self.losses.append(loss)

    def compute_metrics(self) -> Dict[str, float]:
        if not self.all_predictions:
            return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0, 'MSE': 0.0, 'avg_loss': 0.0}
        pa = np.array(self.all_predictions)
        ta = np.array(self.all_targets)
        mse = mean_squared_error(ta, pa)
        return {
            'MAE':      mean_absolute_error(ta, pa),
            'RMSE':     float(np.sqrt(mse)),
            'MAPE':     safe_mape(ta, pa),
            'MSE':      float(mse),
            'avg_loss': float(np.mean(self.losses)) if self.losses else 0.0,
        }


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class LearningRateScheduler:
    def __init__(self, optimizer, mode: str = 'plateau', factor: float = 0.5,
                 patience: int = 5, min_lr: float = 1e-6):
        self.mode = mode
        if mode == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
        elif mode == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=min_lr)
        elif mode == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=patience, gamma=factor)
        else:
            self.scheduler = None

    def step(self, metric: float = None):
        if self.scheduler is None:
            return
        if self.mode == 'plateau' and metric is not None:
            self.scheduler.step(metric)
        elif self.mode in ('cosine', 'step'):
            self.scheduler.step()


class Logger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{experiment_name}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(experiment_name)
        self.metrics_history: Dict = {'train_loss': [], 'val_loss': [], 'test_metrics': {}}

    def log(self, message: str):
        self.logger.info(message)

    def log_metrics(self, epoch: int, train_loss: float,
                    val_loss: float = None, test_metrics: Dict = None):
        self.metrics_history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics_history['val_loss'].append(val_loss)
        if test_metrics is not None:
            self.metrics_history['test_metrics'][epoch] = test_metrics
        msg = f"Epoch {epoch}: Train Loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f", Val Loss={val_loss:.6f}"
        if test_metrics:
            msg += f", MAE={test_metrics.get('MAE', 0):.4f}, RMSE={test_metrics.get('RMSE', 0):.4f}"
        self.log(msg)

    def save_metrics(self):
        np.save(os.path.join(self.log_dir, f'{self.experiment_name}_metrics.npy'),
                self.metrics_history)


class ModelSaver:
    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        os.makedirs(save_dir, exist_ok=True)

    def save_model(self, model: nn.Module, optimizer=None, epoch: int = None,
                   metrics: Dict = None, is_best: bool = False) -> str:
        checkpoint = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'metrics': metrics}
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        parts = [self.experiment_name]
        if epoch is not None:
            parts.append(f'epoch_{epoch}')
        if is_best:
            parts.append('best')
        path = os.path.join(self.save_dir, '_'.join(parts) + '.pth')
        torch.save(checkpoint, path)
        return path

    def load_model(self, model: nn.Module, checkpoint_path: str, optimizer=None):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt.get('epoch', 0), ckpt.get('metrics', {})


class DataVisualizer:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.rcParams.update({
            'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
            'xtick.labelsize': 12, 'ytick.labelsize': 12,
            'legend.fontsize': 12, 'figure.titlesize': 18,
        })

    def _savefig(self, name: str):
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{name}.pdf'),
                    format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_flow_predictions(self, predictions: np.ndarray, targets: np.ndarray,
                              node_indices: List[int] = None, time_indices: List[int] = None,
                              save_name: str = 'flow_predictions_scatter'):
        node_indices = node_indices or list(range(min(4, predictions.shape[1])))
        time_indices = time_indices or list(range(min(6, predictions.shape[2])))
        fig, axes = plt.subplots(len(node_indices), len(time_indices),
                                 figsize=(6 * len(time_indices), 5 * len(node_indices)))
        if len(node_indices) == 1 and len(time_indices) == 1:
            axes = np.array([[axes]])
        elif len(node_indices) == 1:
            axes = axes.reshape(1, -1)
        elif len(time_indices) == 1:
            axes = axes.reshape(-1, 1)

        for i, ni in enumerate(node_indices):
            for j, ti in enumerate(time_indices):
                ax = axes[i, j]
                pv = predictions[:, ni, ti]
                tv = targets[:, ni, ti]
                mask = np.isfinite(pv) & np.isfinite(tv)
                pv, tv = pv[mask], tv[mask]
                if len(pv) > 0:
                    ax.scatter(tv, pv, alpha=0.7, s=40, color='blue',
                               edgecolors='darkblue', linewidth=0.5)
                    lo, hi = min(tv.min(), pv.min()), max(tv.max(), pv.max())
                    ax.plot([lo, hi], [lo, hi], 'r--', lw=3, label='Perfect')
                    mae  = np.mean(np.abs(tv - pv))
                    rmse = np.sqrt(np.mean((tv - pv) ** 2))
                    mape = safe_mape(tv, pv)
                    ax.set_xlabel('True Speed (km/h)', fontweight='bold')
                    ax.set_ylabel('Predicted Speed (km/h)', fontweight='bold')
                    ax.set_title(
                        f'Node {ni}, Horizon {ti+1}\n'
                        f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%',
                        fontweight='bold')
                    ax.grid(True, alpha=0.4)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center',
                            transform=ax.transAxes, fontweight='bold')
                    ax.set_title(f'Node {ni}, Horizon {ti+1}')
        self._savefig(save_name)

    def plot_time_series(self, predictions: np.ndarray, targets: np.ndarray,
                         node_indices: List[int] = None, save_name: str = 'time_series'):
        node_indices = node_indices or list(range(min(6, predictions.shape[1])))
        fig, axes = plt.subplots(len(node_indices), 1,
                                 figsize=(16, 4 * len(node_indices)))
        if len(node_indices) == 1:
            axes = [axes]
        steps = np.arange(1, predictions.shape[2] + 1)
        for i, ni in enumerate(node_indices):
            ax = axes[i]
            ps = predictions[0, ni, :]
            ts = targets[0, ni, :]
            ax.plot(steps, ts, 'b-', lw=3, label='Ground Truth', marker='o', markersize=6)
            ax.plot(steps, ps, 'r--', lw=3, label='Predictions', marker='s', markersize=6)
            mae  = np.mean(np.abs(ts - ps))
            rmse = np.sqrt(np.mean((ts - ps) ** 2))
            ax.set_xlabel('Prediction Horizon', fontweight='bold')
            ax.set_ylabel('Traffic Speed (km/h)', fontweight='bold')
            ax.set_title(f'Node {ni} — MAE: {mae:.2f}, RMSE: {rmse:.2f}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.4)
        self._savefig(save_name)

    def plot_horizon_performance(self, horizon_results: Dict,
                                 save_name: str = 'horizon_performance'):
        horizons = [h for h in horizon_results if h != 'overall']
        if not horizons:
            return
        metrics = ['MAE', 'RMSE', 'MAPE']
        colors  = ['#2E86AB', '#A23B72', '#F18F01']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            vals = [horizon_results[h][metric] for h in horizons]
            bars = axes[idx].bar(horizons, vals, color=color, alpha=0.8, edgecolor='black')
            axes[idx].set_title(f'{metric} Across Horizons', fontweight='bold')
            axes[idx].set_xlabel('Prediction Horizon')
            axes[idx].set_ylabel(metric)
            axes[idx].grid(True, alpha=0.3)
            for bar, val in zip(bars, vals):
                h = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width() / 2.,
                               h * 1.01, f'{val:.3f}', ha='center', va='bottom',
                               fontweight='bold')
        self._savefig(save_name)

    def plot_error_distribution(self, predictions: np.ndarray, targets: np.ndarray,
                                save_name: str = 'error_distribution'):
        errors = (predictions - targets).flatten()
        mask = np.isfinite(errors)
        errors = errors[mask]
        abs_err = np.abs(errors)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(np.mean(errors), color='red',   ls='--', lw=2,
                           label=f'Mean: {np.mean(errors):.3f}')
        axes[0, 0].axvline(np.median(errors), color='green', ls='--', lw=2,
                           label=f'Median: {np.median(errors):.3f}')
        axes[0, 0].set_title('Prediction Error Distribution', fontweight='bold')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(abs_err, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].axvline(np.mean(abs_err), color='red', ls='--', lw=2,
                           label=f'MAE: {np.mean(abs_err):.3f}')
        axes[0, 1].set_title('Absolute Error Distribution', fontweight='bold')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        pct = np.arange(0, 101, 5)
        axes[1, 0].plot(pct, np.percentile(abs_err, pct), 'b-', lw=3, marker='o')
        axes[1, 0].set_title('Error Percentile Analysis', fontweight='bold')
        axes[1, 0].set_xlabel('Percentile'); axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(np.random.normal(0, 1, len(errors)), errors, alpha=0.6)
        axes[1, 1].set_title('Error vs Normal Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Normal Quantiles'); axes[1, 1].set_ylabel('Sample Quantiles')
        axes[1, 1].grid(True, alpha=0.3)

        self._savefig(save_name)

    def plot_temporal_error_patterns(self, predictions: np.ndarray, targets: np.ndarray,
                                     save_name: str = 'temporal_error_patterns'):
        mae_t, rmse_t, mape_t = [], [], []
        for h in range(predictions.shape[2]):
            pv = predictions[:, :, h].flatten()
            tv = targets[:, :, h].flatten()
            mask = np.isfinite(pv) & np.isfinite(tv)
            if mask.sum() > 0:
                mae_t.append(np.mean(np.abs(pv[mask] - tv[mask])))
                rmse_t.append(float(np.sqrt(np.mean((pv[mask] - tv[mask]) ** 2))))
                mape_t.append(safe_mape(tv[mask], pv[mask]))

        steps = list(range(1, len(mae_t) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, vals, label, color, marker in zip(
            axes,
            [mae_t, rmse_t, mape_t],
            ['MAE', 'RMSE', 'MAPE (%)'],
            ['b', 'r', 'g'],
            ['o', 's', '^'],
        ):
            ax.plot(steps, vals, f'{color}-', lw=3, marker=marker, markersize=8)
            ax.set_title(f'{label} vs Prediction Horizon', fontweight='bold')
            ax.set_xlabel('Prediction Horizon'); ax.set_ylabel(label)
            ax.grid(True, alpha=0.3); ax.set_xticks(steps)
        self._savefig(save_name)


def print_model_summary(model: nn.Module, input_shape: Tuple):
    print("=" * 50)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Model Size: {get_model_size(model):.2f} MB")
    print(f"Input Shape: {input_shape}")
    print("=" * 50)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.2f}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {seconds % 60:.2f}s"


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time   = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        return self.elapsed()

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        return False


def validate_config(config: Dict) -> bool:
    return all(k in config for k in ('data', 'model', 'training', 'few_shot'))


def create_experiment_name(config: Dict) -> str:
    return (f"{config['model']['name']}_"
            f"{config['training']['test_dataset']}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}")


def get_metrics_summary(metrics: Dict[str, float]) -> str:
    parts = [f"{m}: {metrics[m]:.4f}" for m in ('MAE', 'RMSE', 'MAPE') if m in metrics]
    return " | ".join(parts) if parts else "No metrics available"