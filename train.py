import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict
from typing import Dict, List, Optional

from torch_geometric.data import Data

from models.model import MCPST_FSL, PhysicsInformedLoss
from datasets import TrafficDataManager, TrafficDataset
from utils import MetricsCalculator, EarlyStopping, LearningRateScheduler, Timer, format_time


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _default_minibatch_size() -> int:
    """Return the device-appropriate default minibatch size."""
    return 2048 if torch.cuda.is_available() else 4096


def _slice_batch(data: Data, start: int, end: int) -> Data:
    """
    Return a lightweight slice of a PyG Data batch along the sample (first)
    dimension.

    ``data.y`` is a dict in this codebase (keys: 'flow', 'spatial',
    'temporal'), so we slice each value independently.  Graph topology
    (edge_index) is shared across all samples from the same dataset and is
    therefore *not* copied.
    """
    sliced = Data()
    sliced.x = data.x[start:end]
    if isinstance(data.y, dict):
        sliced.y = {k: v[start:end] for k, v in data.y.items()}
    else:
        sliced.y = data.y[start:end]
    sliced.node_num = data.node_num
    sliced.edge_index = data.edge_index
    if hasattr(data, 'data_name'):
        sliced.data_name = data.data_name
    return sliced


# ---------------------------------------------------------------------------
# Few-shot trainer
# ---------------------------------------------------------------------------

class EnhancedFewShotTrainer:
    def __init__(self, model: MCPST_FSL, config: Dict, device: torch.device, logger):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger

        self.adaptation_steps = int(config['few_shot']['adaptation_steps'])
        self.criterion = PhysicsInformedLoss(config)

        # ------------------------------------------------------------------
        # minibatch_size: maximum samples processed in a single forward/
        # backward pass.  Inner-loop support sets and outer-loop query sets
        # are both chunked using this value.
        # ------------------------------------------------------------------
        self.minibatch_size = int(
            config.get('training', {}).get('minibatch_size', _default_minibatch_size())
        )

        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['few_shot'].get('outer_lr', 0.0001)),
            weight_decay=float(config['training']['weight_decay']),
        )
        self.scheduler = LearningRateScheduler(
            self.meta_optimizer,
            mode=config['training']['scheduler']['type'],
            factor=float(config['training']['scheduler']['factor']),
            patience=int(config['training']['scheduler']['patience']),
            min_lr=float(config['training']['scheduler']['min_lr']),
        )
        self.metrics_calculator = MetricsCalculator()
        self.early_stopping = EarlyStopping(
            patience=int(config['training'].get('early_stopping_patience', 20)),
            min_delta=float(config['training'].get('early_stopping_min_delta', 1e-5)),
            restore_best=True,
        )
        self.timer = Timer()

    # ------------------------------------------------------------------
    # Inner-loop (fast) adaptation — chunked over support minibatches
    # ------------------------------------------------------------------

    def inner_adaptation(self, support_data: Data, support_adjacency: torch.Tensor) -> nn.Module:
        """
        Run ``adaptation_steps`` gradient steps on the support set.

        The support set is split into minibatches of at most
        ``self.minibatch_size`` samples; gradients are accumulated before
        each inner-loop optimiser step so the effective update is identical
        to processing the full support set at once.
        """
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=float(self.config['few_shot'].get('inner_lr', 0.0005)),
        )

        batch_n = support_data.x.shape[0]
        num_chunks = max(1, math.ceil(batch_n / self.minibatch_size))
        adjacency = support_adjacency.to(self.device)

        for _ in range(self.adaptation_steps):
            inner_optimizer.zero_grad()

            for chunk_start in range(0, batch_n, self.minibatch_size):
                chunk_end = min(chunk_start + self.minibatch_size, batch_n)
                mb = _slice_batch(support_data, chunk_start, chunk_end)
                chunk_weight = (chunk_end - chunk_start) / batch_n

                x = mb.x.to(self.device)
                if x.dim() == 3:
                    x = x.unsqueeze(1)

                outputs = adapted_model(x, adjacency)
                targets = {k: v.to(self.device) for k, v in mb.y.items()}
                loss_dict = self.criterion(outputs, targets)

                # Scale so each sample contributes equally regardless of
                # chunk size; divide by num_chunks to keep gradient magnitude
                # independent of the number of chunks.
                loss = loss_dict['total_loss'] * chunk_weight
                if torch.isfinite(loss):
                    loss.backward()

            torch.nn.utils.clip_grad_norm_(
                adapted_model.parameters(),
                float(self.config['few_shot'].get('gradient_clip_inner', 1.0)),
            )
            inner_optimizer.step()

        return adapted_model

    # ------------------------------------------------------------------
    # Outer-loop (meta) update — chunked over query minibatches
    # ------------------------------------------------------------------

    def meta_update(self, support_tasks, support_adjacencies,
                    query_tasks, query_adjacencies):
        meta_losses: List[torch.Tensor] = []
        loss_components: Dict[str, List] = defaultdict(list)
        # Keep references so we can transfer FOMAML grads after backward.
        adapted_models: List[nn.Module] = []

        for support_data, support_adj, query_data, query_adj in zip(
            support_tasks, support_adjacencies, query_tasks, query_adjacencies
        ):
            adapted_model = self.inner_adaptation(support_data, support_adj)
            adapted_model.train()   # train mode so BN/dropout behave consistently
            adapted_models.append(adapted_model)

            query_adj_dev = query_adj.to(self.device)
            batch_n = query_data.x.shape[0]
            num_chunks = max(1, math.ceil(batch_n / self.minibatch_size))

            task_loss: Optional[torch.Tensor] = None
            task_components: Dict[str, float] = defaultdict(float)

            for chunk_start in range(0, batch_n, self.minibatch_size):
                chunk_end = min(chunk_start + self.minibatch_size, batch_n)
                mb = _slice_batch(query_data, chunk_start, chunk_end)
                chunk_weight = (chunk_end - chunk_start) / batch_n

                x = mb.x.to(self.device)
                if x.dim() == 3:
                    x = x.unsqueeze(1)

                outputs = adapted_model(x, query_adj_dev)
                targets = {k: v.to(self.device) for k, v in mb.y.items()}
                loss_dict = self.criterion(outputs, targets)

                chunk_loss = loss_dict['total_loss'] * chunk_weight
                task_loss = chunk_loss if task_loss is None else task_loss + chunk_loss

                for component, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        task_components[component] += value.item() * chunk_weight
                    else:
                        task_components[component] += float(value) * chunk_weight

            if task_loss is not None:
                meta_losses.append(task_loss)
            for k, v in task_components.items():
                loss_components[k].append(v)

        if not meta_losses:
            return {'meta_loss': 0.0}

        meta_loss = torch.stack(meta_losses).mean()
        self.meta_optimizer.zero_grad()

        if torch.isfinite(meta_loss):
            meta_loss.backward()

            # ── FOMAML gradient transfer ──────────────────────────────────────
            # Match by *name*, not by position.  zip(model.parameters(),
            # adapted.parameters()) fails when adapted_model ran on a
            # different dataset: _get_or_create_aggregator dynamically adds
            # node-count-specific Linear layers, making parameter counts and
            # orderings diverge between models.  Named matching is robust:
            # dynamic-only or shape-mismatched params are silently skipped.
            n_tasks = max(len(adapted_models), 1)
            for p in self.model.parameters():
                p.grad = None
            meta_named = dict(self.model.named_parameters())
            for adapted in adapted_models:
                for name, p_adpt in adapted.named_parameters():
                    if p_adpt.grad is None:
                        continue
                    if name not in meta_named:
                        continue  # dynamic param absent from meta-model
                    p_meta = meta_named[name]
                    if p_meta.shape != p_adpt.shape:
                        continue  # node-count-specific layer; skip
                    g = p_adpt.grad.detach() / n_tasks
                    if p_meta.grad is None:
                        p_meta.grad = g.clone()
                    else:
                        p_meta.grad.add_(g)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config['few_shot'].get('gradient_clip_outer', 1.0)),
            )
            self.meta_optimizer.step()

        return {
            'meta_loss': meta_loss.item(),
            **{
                c: float(np.mean(v)) if v else 0.0
                for c, v in loss_components.items()
                if c != 'total_loss'
            },
        }

    # ------------------------------------------------------------------
    # Evaluation on target domain — chunked query forward pass
    # ------------------------------------------------------------------

    def evaluate_on_target(self, target_dataset: TrafficDataset, num_episodes: int = 5):
        self.model.eval()
        self.metrics_calculator.reset()
        total_loss = 0.0
        valid_episodes = 0

        # Denormalisation stats — needed for meaningful MAPE on physical units.
        # Without this, MAPE is computed on normalised values (mean≈0, std≈1)
        # where any small value produces enormous percentage errors (~100%+).
        _ds_name   = target_dataset.data_list[0]
        _dn_mean   = float(target_dataset.means_list.get(_ds_name, 0.0))
        _dn_std    = float(target_dataset.stds_list.get(_ds_name, 1.0))

        for episode in range(num_episodes):
            support_tasks, support_adjs, query_tasks, query_adjs = \
                target_dataset.get_few_shot_tasks(
                    1,
                    self.config['few_shot']['support_size'],
                    self.config['few_shot']['query_size'],
                )
            if not support_tasks or not query_tasks:
                continue

            adapted_model = self.inner_adaptation(support_tasks[0], support_adjs[0])
            adapted_model.eval()

            query_adj_dev = query_adjs[0].to(self.device)
            batch_n = query_tasks[0].x.shape[0]

            episode_preds: List[torch.Tensor] = []
            episode_targets: List[torch.Tensor] = []
            episode_loss = 0.0

            with torch.no_grad():
                for chunk_start in range(0, batch_n, self.minibatch_size):
                    chunk_end = min(chunk_start + self.minibatch_size, batch_n)
                    mb = _slice_batch(query_tasks[0], chunk_start, chunk_end)
                    chunk_weight = (chunk_end - chunk_start) / batch_n

                    x = mb.x.to(self.device)
                    if x.dim() == 3:
                        x = x.unsqueeze(1)

                    outputs = adapted_model(x, query_adj_dev)
                    targets = {k: v.to(self.device) for k, v in mb.y.items()}
                    loss_dict = self.criterion(outputs, targets)

                    episode_loss += loss_dict['total_loss'].item() * chunk_weight
                    episode_preds.append(outputs['flow_predictions'])

                    flow_t = targets.get('flow', targets.get('y', None))
                    if flow_t is not None:
                        if flow_t.dim() == 4:
                            flow_t = flow_t[..., 0]
                        episode_targets.append(flow_t)

            if not episode_preds or not episode_targets:
                continue

            preds_cat = torch.cat(episode_preds, dim=0)
            tgts_cat = torch.cat(episode_targets, dim=0)

            if preds_cat.dim() != tgts_cat.dim():
                continue

            min_b = min(preds_cat.shape[0], tgts_cat.shape[0])
            if preds_cat.dim() >= 3 and tgts_cat.dim() >= 3:
                min_nodes = min(preds_cat.shape[1], tgts_cat.shape[1])
                min_time = min(preds_cat.shape[2], tgts_cat.shape[2])
                p = preds_cat[:min_b, :min_nodes, :min_time]
                t = tgts_cat[:min_b, :min_nodes, :min_time]
            else:
                p = preds_cat[:min_b]
                t = tgts_cat[:min_b]

            if p.shape == t.shape and p.numel() > 0:
                # Denormalise to physical units before computing metrics so
                # that MAE is in km/h and MAPE is not inflated by near-zero
                # normalised values.
                p_np = np.maximum(p.detach().cpu().numpy() * _dn_std + _dn_mean, 0.1)
                t_np = np.maximum(t.detach().cpu().numpy() * _dn_std + _dn_mean, 0.1)
                self.metrics_calculator.update(p_np, t_np, episode_loss)
                total_loss += episode_loss
                valid_episodes += 1

        if valid_episodes == 0:
            return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0, 'episode_loss': 0.0}

        metrics = self.metrics_calculator.compute_metrics()
        metrics['episode_loss'] = total_loss / valid_episodes
        return metrics

    # ------------------------------------------------------------------
    # Full two-phase training loop
    # ------------------------------------------------------------------

    def train(self, data_manager: TrafficDataManager):
        self.timer.start()
        self.logger.log("Starting MCPST-FSL two-phase meta-learning training")
        self.logger.log(f"Minibatch size: {self.minibatch_size}")

        source_dataset = data_manager.create_dataset(
            'source', self.config['training']['test_dataset'],
            add_physics_features=self.config['training']['add_physics_features'],
            minibatch_size=self.minibatch_size,
        )
        target_dataset = data_manager.create_dataset(
            'target', self.config['training']['test_dataset'],
            target_days=self.config['training']['target_days'],
            add_physics_features=self.config['training']['add_physics_features'],
            minibatch_size=self.minibatch_size,
        )

        training_history = {
            k: [] for k in [
                'meta_loss', 'flow_loss', 'spatial_loss', 'temporal_loss',
                'physics_loss', 'uncertainty_loss', 'val_metrics',
            ]
        }

        # ---- Phase 1: source-domain meta-learning ----
        source_epochs = min(int(self.config['training']['source_epochs']), 300)
        self.logger.log(f"PHASE 1: Source domain meta-learning ({source_epochs} epochs)")
        print("=" * 60)
        print(f"PHASE 1: SOURCE DOMAIN META-LEARNING ({source_epochs} epochs)")
        print("=" * 60)

        for epoch in range(source_epochs):
            epoch_losses = {k: [] for k in [
                'meta_loss', 'flow_loss', 'spatial_loss',
                'temporal_loss', 'physics_loss', 'uncertainty_loss',
            ]}

            for meta_step in range(10):
                support_tasks, support_adjs, query_tasks, query_adjs = \
                    source_dataset.get_few_shot_tasks(
                        int(self.config['few_shot']['num_tasks']),
                        int(self.config['few_shot']['support_size']),
                        int(self.config['few_shot']['query_size']),
                    )
                if support_tasks and query_tasks:
                    step_losses = self.meta_update(
                        support_tasks, support_adjs, query_tasks, query_adjs
                    )
                    for key, value in step_losses.items():
                        if key in epoch_losses:
                            epoch_losses[key].append(value)

            avg_losses = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
            for k, v in avg_losses.items():
                if k in training_history:
                    training_history[k].append(v)

            if epoch % 10 == 0 or epoch == source_epochs - 1:
                val_metrics = self.evaluate_on_target(target_dataset, 5)
                training_history['val_metrics'].append(val_metrics)
                mae = val_metrics.get('MAE', 0.0)
                mape = val_metrics.get('MAPE', 0.0)

                print(f"Source Epoch {epoch:3d}/{source_epochs} | "
                      f"Meta: {avg_losses.get('meta_loss', 0):.4f} | "
                      f"Flow: {avg_losses.get('flow_loss', 0):.4f} | "
                      f"Spatial: {avg_losses.get('spatial_loss', 0):.4f} | "
                      f"Temporal: {avg_losses.get('temporal_loss', 0):.4f} | "
                      f"Uncertainty: {avg_losses.get('uncertainty_loss', 0):.4f} | "
                      f"MAE: {mae:.4f} | MAPE: {mape:.2f}%")

                self.logger.log_metrics(
                    epoch, avg_losses.get('meta_loss', 0.0),
                    val_metrics['episode_loss'], val_metrics
                )
                self.scheduler.step(val_metrics['episode_loss'])

                if self.early_stopping(val_metrics['episode_loss'], self.model):
                    elapsed = format_time(self.timer.elapsed())
                    print(f"Early stopping at epoch {epoch}. Time: {elapsed}")
                    self.logger.log(f"Early stopping at epoch {epoch}. Time: {elapsed}")
                    break
            else:
                print(f"Source Epoch {epoch:3d}/{source_epochs} | "
                      f"Meta: {avg_losses.get('meta_loss', 0):.4f} | "
                      f"Flow: {avg_losses.get('flow_loss', 0):.4f} | "
                      f"Uncertainty: {avg_losses.get('uncertainty_loss', 0):.4f}")

        # ---- Phase 2: target-domain adaptation ----
        target_epochs = min(int(self.config['training']['target_epochs']), 300)
        target_lr = float(self.config['training'].get('target_lr', 0.0003))

        self.logger.log(f"PHASE 2: Target domain adaptation ({target_epochs} epochs)")
        print("\n" + "=" * 60)
        print(f"PHASE 2: TARGET DOMAIN ADAPTATION ({target_epochs} epochs, lr={target_lr})")
        print("=" * 60)

        if target_epochs > 0:
            target_optimizer = torch.optim.Adam(
                self.model.parameters(), lr=target_lr,
                weight_decay=float(self.config['training']['weight_decay']),
            )

            for epoch in range(target_epochs):
                epoch_loss = 0.0
                num_updates = 0

                for _ in range(8):
                    support_tasks, support_adjs, query_tasks, query_adjs = \
                        target_dataset.get_few_shot_tasks(
                            1,
                            int(self.config['few_shot']['support_size']),
                            int(self.config['few_shot']['query_size']),
                        )
                    if not support_tasks or not query_tasks:
                        continue

                    adapted_model = self.inner_adaptation(support_tasks[0], support_adjs[0])
                    adapted_model.train()

                    target_optimizer.zero_grad()
                    query_adj_dev = query_adjs[0].to(self.device)
                    batch_n = query_tasks[0].x.shape[0]
                    step_loss = 0.0

                    for chunk_start in range(0, batch_n, self.minibatch_size):
                        chunk_end = min(chunk_start + self.minibatch_size, batch_n)
                        mb = _slice_batch(query_tasks[0], chunk_start, chunk_end)
                        chunk_weight = (chunk_end - chunk_start) / batch_n

                        x = mb.x.to(self.device)
                        if x.dim() == 3:
                            x = x.unsqueeze(1)

                        outputs = adapted_model(x, query_adj_dev)
                        targets_dict = {k: v.to(self.device) for k, v in mb.y.items()}
                        loss_dict = self.criterion(outputs, targets_dict)
                        loss = loss_dict['total_loss'] * chunk_weight

                        if torch.isfinite(loss):
                            loss.backward()
                            step_loss += loss.item()

                    # FOMAML: transfer grads from adapted_model → self.model.
                    # Use named matching to handle dynamically-created layers
                    # (node-count-specific aggregators) that differ in shape.
                    for p in self.model.parameters():
                        p.grad = None
                    meta_named = dict(self.model.named_parameters())
                    for name, p_adpt in adapted_model.named_parameters():
                        if p_adpt.grad is None:
                            continue
                        if name not in meta_named:
                            continue
                        p_meta = meta_named[name]
                        if p_meta.shape != p_adpt.shape:
                            continue
                        p_meta.grad = p_adpt.grad.detach().clone()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(self.config['few_shot'].get('gradient_clip_outer', 1.0)),
                    )
                    target_optimizer.step()
                    epoch_loss += step_loss
                    num_updates += 1

                avg_loss = epoch_loss / max(num_updates, 1)

                if epoch % 15 == 0 or epoch == target_epochs - 1:
                    val_metrics = self.evaluate_on_target(target_dataset, 3)
                    mae = val_metrics.get('MAE', 0.0)
                    rmse = val_metrics.get('RMSE', 0.0)
                    mape = val_metrics.get('MAPE', 0.0)
                    print(f"Target Epoch {epoch:3d}/{target_epochs} | "
                          f"Loss: {avg_loss:.4f} | MAE: {mae:.4f} | "
                          f"RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
                else:
                    print(f"Target Epoch {epoch:3d}/{target_epochs} | Loss: {avg_loss:.4f}")

        elapsed_str = format_time(self.timer.stop())
        self.logger.log(f"MCPST-FSL training completed in {elapsed_str}")
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETED. Total Time: {elapsed_str}")
        print("=" * 60)

        return training_history


# ---------------------------------------------------------------------------
# Standard trainer
# ---------------------------------------------------------------------------

class EnhancedStandardTrainer:
    def __init__(self, model: MCPST_FSL, config: Dict, device: torch.device, logger):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger

        # ------------------------------------------------------------------
        # minibatch_size: maximum samples processed per forward/backward pass.
        # train_step uses gradient accumulation so the effective update is
        # equivalent to processing the whole batch at once.
        # ------------------------------------------------------------------
        self.minibatch_size = int(
            config.get('training', {}).get('minibatch_size', _default_minibatch_size())
        )

        self.criterion = PhysicsInformedLoss(config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['training'].get('source_lr', 0.0003)),
            weight_decay=float(config['training']['weight_decay']),
        )
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            mode=config['training']['scheduler']['type'],
            factor=float(config['training']['scheduler']['factor']),
            patience=int(config['training']['scheduler']['patience']),
            min_lr=float(config['training']['scheduler']['min_lr']),
        )
        self.metrics_calculator = MetricsCalculator()
        self.early_stopping = EarlyStopping(
            patience=int(config['training'].get('early_stopping_patience', 20)),
            min_delta=float(config['training'].get('early_stopping_min_delta', 1e-5)),
            restore_best=True,
        )
        self.timer = Timer()

    # ------------------------------------------------------------------
    # Single training step with gradient accumulation
    # ------------------------------------------------------------------

    def train_step(self, data: Data, adjacency: torch.Tensor) -> Dict[str, float]:
        """
        Process one loaded batch using minibatch gradient accumulation.

        ``data.x.shape[0]`` may be larger than ``self.minibatch_size`` (when
        the source dataset draws large samples).  Each chunk contributes
        proportionally (weighted by chunk_size / batch_size) so the resulting
        gradient is equivalent to processing the full batch at once.
        """
        self.model.train()

        batch_n = data.x.shape[0]
        adjacency = adjacency.to(self.device)
        self.optimizer.zero_grad()

        accum: Dict[str, float] = defaultdict(float)

        for chunk_start in range(0, batch_n, self.minibatch_size):
            chunk_end = min(chunk_start + self.minibatch_size, batch_n)
            mb = _slice_batch(data, chunk_start, chunk_end)
            chunk_weight = (chunk_end - chunk_start) / batch_n

            x = mb.x.to(self.device)
            if x.dim() == 3:
                x = x.unsqueeze(1)

            outputs = self.model(x, adjacency)
            targets = {k: v.to(self.device) for k, v in mb.y.items()}
            loss_dict = self.criterion(outputs, targets)

            loss = loss_dict['total_loss'] * chunk_weight
            if torch.isfinite(loss):
                loss.backward()

            for k, v in loss_dict.items():
                accum[k] += (v.item() if isinstance(v, torch.Tensor) else float(v)) * chunk_weight

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            float(self.config['training'].get('gradient_clip', 1.0)),
        )
        self.optimizer.step()

        return dict(accum)

    # ------------------------------------------------------------------
    # Evaluation — chunked forward pass, no gradients
    # ------------------------------------------------------------------

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        self.metrics_calculator.reset()
        total_losses: List[float] = []

        # Denormalisation stats for meaningful MAPE.
        ds = getattr(dataloader, 'dataset', None)
        if ds is not None and hasattr(ds, 'data_list') and hasattr(ds, 'means_list'):
            _ds_name = ds.data_list[0]
            _dn_mean = float(ds.means_list.get(_ds_name, 0.0))
            _dn_std  = float(ds.stds_list.get(_ds_name, 1.0))
        else:
            _dn_mean, _dn_std = 0.0, 1.0

        with torch.no_grad():
            for batch_idx, (data, adjacency) in enumerate(dataloader):
                adjacency = adjacency.to(self.device)
                batch_n = data.x.shape[0]

                batch_preds: List[torch.Tensor] = []
                batch_targets: List[torch.Tensor] = []
                batch_loss = 0.0

                for chunk_start in range(0, batch_n, self.minibatch_size):
                    chunk_end = min(chunk_start + self.minibatch_size, batch_n)
                    mb = _slice_batch(data, chunk_start, chunk_end)
                    chunk_weight = (chunk_end - chunk_start) / batch_n

                    x = mb.x.to(self.device)
                    if x.dim() == 3:
                        x = x.unsqueeze(1)

                    outputs = self.model(x, adjacency)
                    targets = {k: v.to(self.device) for k, v in mb.y.items()}
                    loss_dict = self.criterion(outputs, targets)

                    batch_loss += loss_dict['total_loss'].item() * chunk_weight
                    batch_preds.append(outputs['flow_predictions'])

                    flow_t = targets.get('flow', targets.get('y', None))
                    if flow_t is not None:
                        if flow_t.dim() == 4:
                            flow_t = flow_t[..., 0]
                        batch_targets.append(flow_t)

                if batch_preds and batch_targets:
                    preds_cat = torch.cat(batch_preds, dim=0)
                    tgts_cat = torch.cat(batch_targets, dim=0)
                    min_nodes = min(preds_cat.shape[1], tgts_cat.shape[1])
                    min_time = min(preds_cat.shape[2], tgts_cat.shape[2])
                    p = preds_cat[:, :min_nodes, :min_time]
                    t = tgts_cat[:, :min_nodes, :min_time]
                    if p.shape == t.shape:
                        p_np = np.maximum(p.detach().cpu().numpy() * _dn_std + _dn_mean, 0.1)
                        t_np = np.maximum(t.detach().cpu().numpy() * _dn_std + _dn_mean, 0.1)
                        self.metrics_calculator.update(p_np, t_np, batch_loss)

                total_losses.append(batch_loss)
                if batch_idx >= 20:
                    break

        metrics = self.metrics_calculator.compute_metrics()
        metrics['total_loss'] = float(np.mean(total_losses)) if total_losses else 0.0
        return metrics

    # ------------------------------------------------------------------
    # Full two-phase training loop
    # ------------------------------------------------------------------

    def train(self, dataloaders: Dict) -> Dict:
        self.timer.start()
        self.logger.log("Starting MCPST-FSL two-phase standard training")
        self.logger.log(f"Minibatch size: {self.minibatch_size}")

        train_loader = dataloaders['source']
        val_loader = dataloaders['target']
        training_history: Dict[str, List] = {
            'train_loss': [], 'val_loss': [], 'val_metrics': []
        }

        # ---- Phase 1: source-domain training ----
        source_epochs = min(int(self.config['training']['source_epochs']), 300)
        self.logger.log(f"PHASE 1: Source domain training ({source_epochs} epochs)")
        print("=" * 60)
        print(f"PHASE 1: SOURCE DOMAIN TRAINING ({source_epochs} epochs)")
        print("=" * 60)

        for epoch in range(source_epochs):
            epoch_losses: List[float] = []

            for batch_idx, (data, adjacency) in enumerate(train_loader):
                step_losses = self.train_step(data, adjacency)
                epoch_losses.append(step_losses.get('total_loss', 0.0))
                if batch_idx >= 25:
                    break

            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            training_history['train_loss'].append(avg_train_loss)

            if epoch % 10 == 0 or epoch == source_epochs - 1:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['total_loss']
                training_history['val_loss'].append(val_loss)
                training_history['val_metrics'].append(val_metrics)
                mae = val_metrics.get('MAE', 0.0)
                rmse = val_metrics.get('RMSE', 0.0)
                mape = val_metrics.get('MAPE', 0.0)

                print(f"Source Epoch {epoch:3d}/{source_epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
                self.logger.log_metrics(epoch, avg_train_loss, val_loss, val_metrics)
                self.scheduler.step(val_loss)

                if self.early_stopping(val_loss, self.model):
                    elapsed = format_time(self.timer.elapsed())
                    print(f"Early stopping at epoch {epoch}. Time: {elapsed}")
                    self.logger.log(f"Early stopping at epoch {epoch}. Time: {elapsed}")
                    break
            else:
                print(f"Source Epoch {epoch:3d}/{source_epochs} | Train Loss: {avg_train_loss:.6f}")

        # ---- Phase 2: target-domain fine-tuning ----
        target_epochs = min(int(self.config['training']['target_epochs']), 300)
        target_lr = float(self.config['training'].get('target_lr', 0.0001))

        self.logger.log(f"PHASE 2: Target domain fine-tuning ({target_epochs} epochs)")
        print("\n" + "=" * 60)
        print(f"PHASE 2: TARGET DOMAIN FINE-TUNING ({target_epochs} epochs, lr={target_lr})")
        print("=" * 60)

        if target_epochs > 0:
            target_optimizer = torch.optim.Adam(
                self.model.parameters(), lr=target_lr,
                weight_decay=float(self.config['training']['weight_decay']),
            )

            for epoch in range(target_epochs):
                epoch_losses_ft: List[float] = []

                for batch_idx, (data, adjacency) in enumerate(val_loader):
                    self.model.train()
                    adjacency = adjacency.to(self.device)
                    batch_n = data.x.shape[0]
                    target_optimizer.zero_grad()
                    step_loss = 0.0

                    for chunk_start in range(0, batch_n, self.minibatch_size):
                        chunk_end = min(chunk_start + self.minibatch_size, batch_n)
                        mb = _slice_batch(data, chunk_start, chunk_end)
                        chunk_weight = (chunk_end - chunk_start) / batch_n

                        x = mb.x.to(self.device)
                        if x.dim() == 3:
                            x = x.unsqueeze(1)

                        outputs = self.model(x, adjacency)
                        targets_dict = {k: v.to(self.device) for k, v in mb.y.items()}
                        loss_dict = self.criterion(outputs, targets_dict)
                        loss = loss_dict['total_loss'] * chunk_weight

                        if torch.isfinite(loss):
                            loss.backward()
                            step_loss += loss.item()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(self.config['training'].get('gradient_clip', 1.0)),
                    )
                    target_optimizer.step()
                    epoch_losses_ft.append(step_loss)

                    if batch_idx >= 15:
                        break

                avg_loss = float(np.mean(epoch_losses_ft)) if epoch_losses_ft else 0.0

                if epoch % 15 == 0 or epoch == target_epochs - 1:
                    val_metrics = self.evaluate(val_loader)
                    mae = val_metrics.get('MAE', 0.0)
                    rmse = val_metrics.get('RMSE', 0.0)
                    mape = val_metrics.get('MAPE', 0.0)
                    print(f"Target Epoch {epoch:3d}/{target_epochs} | "
                          f"Loss: {avg_loss:.6f} | MAE: {mae:.4f} | "
                          f"RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
                else:
                    print(f"Target Epoch {epoch:3d}/{target_epochs} | Loss: {avg_loss:.6f}")

        elapsed_str = format_time(self.timer.stop())
        self.logger.log(f"MCPST-FSL training completed in {elapsed_str}")
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETED. Total Time: {elapsed_str}")
        print("=" * 60)

        return training_history


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_trainer(model: MCPST_FSL, config: Dict, device: torch.device, logger):
    if config['few_shot']['enabled']:
        return EnhancedFewShotTrainer(model, config, device, logger)
    return EnhancedStandardTrainer(model, config, device, logger)