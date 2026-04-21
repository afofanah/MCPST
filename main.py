import torch
import numpy as np
import os
import argparse
from typing import Dict

from models.model import MCPST_FSL, PhysicsInformedLoss
from datasets import TrafficDataManager
from train import create_trainer
from utils import (
    set_seed, get_device, load_config, save_config,
    Logger, ModelSaver, DataVisualizer,
    print_model_summary, create_experiment_name, validate_config,
    Timer, format_time, get_metrics_summary, safe_mape,
    denormalize_data
)


def _default_minibatch_size() -> int:
    """Return the device-appropriate default minibatch size."""
    return 2048 if torch.cuda.is_available() else 4096


def setup_experiment(config: Dict):
    experiment_name = create_experiment_name(config)
    for directory in [
        config['experiment']['save_dir'],
        config['experiment']['log_dir'],
        config['experiment']['plot_dir'],
    ]:
        os.makedirs(directory, exist_ok=True)
    logger = Logger(config['experiment']['log_dir'], experiment_name)
    model_saver = ModelSaver(config['experiment']['save_dir'], experiment_name)
    visualizer = DataVisualizer(config['experiment']['plot_dir'])
    return experiment_name, logger, model_saver, visualizer


def create_model(config: Dict, device: torch.device) -> MCPST_FSL:
    config['model']['input_dim'] = 6 if config['training']['add_physics_features'] else 2
    return MCPST_FSL(config).to(device)


def evaluate_horizons_fixed(model: MCPST_FSL, test_loader, device: torch.device,
                             config: Dict, data_manager: TrafficDataManager,
                             logger: Logger, minibatch_size: int = 4096):
    """
    Run horizon-specific evaluation on the test set.

    Each DataLoader batch is split into minibatches of at most
    ``minibatch_size`` samples so that large test batches (set to
    minibatch_size in create_dataloaders) stay within memory bounds.
    """
    model.eval()
    all_predictions = []
    all_targets = []

    dataset_name = config['training']['test_dataset']
    dataset_stats = data_manager.get_statistics()
    test_key = f"test_{dataset_name}"

    if test_key in dataset_stats:
        speed_mean = dataset_stats[test_key]['mean']
        speed_std = dataset_stats[test_key]['std']
    else:
        speed_mean = config['data'][dataset_name].get('speed_mean', 0.0)
        speed_std = config['data'][dataset_name].get('speed_std', 1.0)

    logger.log(f"Normalization stats — mean: {speed_mean:.4f}, std: {speed_std:.4f}")

    with torch.no_grad():
        for batch_idx, (data, adjacency) in enumerate(test_loader):
            adjacency = adjacency.to(device)
            batch_n = data.x.shape[0]

            batch_preds = []
            batch_targets_list = []

            # ----------------------------------------------------------
            # Chunk the loaded batch into minibatches; the test DataLoader
            # already uses minibatch_size as its batch_size so in practice
            # there is usually just one chunk — but the loop keeps things
            # safe if the last batch happens to be smaller.
            # ----------------------------------------------------------
            for chunk_start in range(0, batch_n, minibatch_size):
                chunk_end = min(chunk_start + minibatch_size, batch_n)

                x_chunk = data.x[chunk_start:chunk_end].to(device)
                if x_chunk.dim() == 3:
                    x_chunk = x_chunk.unsqueeze(1)

                y_chunk = data.y
                if isinstance(y_chunk, dict):
                    flow_targets_chunk = y_chunk.get(
                        'flow', y_chunk.get('y', list(y_chunk.values())[0])
                    )
                    if isinstance(flow_targets_chunk, torch.Tensor):
                        flow_targets_chunk = flow_targets_chunk[chunk_start:chunk_end]
                else:
                    flow_targets_chunk = y_chunk[chunk_start:chunk_end]

                outputs = model(x_chunk, adjacency)
                predictions = outputs['flow_predictions']

                if isinstance(flow_targets_chunk, torch.Tensor):
                    flow_targets_chunk = flow_targets_chunk.to(device)
                if flow_targets_chunk.dim() == 4:
                    flow_targets_chunk = flow_targets_chunk[..., 0]

                min_n = min(predictions.shape[0], flow_targets_chunk.shape[0])
                min_nodes = min(predictions.shape[1], flow_targets_chunk.shape[1])
                min_steps = min(predictions.shape[2], flow_targets_chunk.shape[2])

                predictions = predictions[:min_n, :min_nodes, :min_steps]
                flow_targets_chunk = flow_targets_chunk[:min_n, :min_nodes, :min_steps]

                pred_denorm = np.maximum(
                    denormalize_data(predictions, speed_mean, speed_std), 0.1
                )
                tgt_denorm = np.maximum(
                    denormalize_data(flow_targets_chunk, speed_mean, speed_std), 0.1
                )
                batch_preds.append(pred_denorm)
                batch_targets_list.append(tgt_denorm)

            if batch_preds:
                all_predictions.append(np.concatenate(batch_preds, axis=0))
                all_targets.append(np.concatenate(batch_targets_list, axis=0))

            if batch_idx >= 20:
                break

    predictions_array = np.concatenate(all_predictions, axis=0)
    targets_array = np.concatenate(all_targets, axis=0)

    num_horizons = min(predictions_array.shape[2], 6)
    horizon_names = (
        ['5min', '15min', '30min', '60min', '90min', '120min']
        if dataset_name.lower() in ['metr-la', 'pems-bay']
        else ['10min', '20min', '30min', '60min', '90min', '120min']
    )

    horizon_results = {}
    for i, horizon_name in enumerate(horizon_names[:num_horizons]):
        pred_flat = predictions_array[:, :, i].flatten()
        target_flat = targets_array[:, :, i].flatten()
        valid_mask = (
            np.isfinite(pred_flat) & np.isfinite(target_flat)
            & (target_flat > 0.1) & (pred_flat > 0)
        )

        if valid_mask.sum() > 0:
            pv = pred_flat[valid_mask]
            tv = target_flat[valid_mask]
            mae = np.mean(np.abs(tv - pv))
            mse = np.mean((tv - pv) ** 2)
            rmse = np.sqrt(mse)
            mape = safe_mape(tv, pv)
        else:
            mae = rmse = mape = mse = 999.0

        horizon_results[horizon_name] = {
            'MAE': float(mae), 'RMSE': float(rmse),
            'MAPE': float(mape), 'MSE': float(mse),
        }

    logger.log("=" * 50)
    logger.log("EVALUATION RESULTS (DENORMALIZED)")
    logger.log("=" * 50)
    for h in horizon_names[:num_horizons]:
        m = horizon_results[h]
        logger.log(
            f"Horizon {h}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, MAPE={m['MAPE']:.2f}%"
        )

    return horizon_results, predictions_array, targets_array


def print_horizon_results(horizon_results: Dict, dataset_name: str, logger: Logger):
    print("\n" + "=" * 80)
    print(f"PREDICTION HORIZON RESULTS - {dataset_name.upper()}")
    print("=" * 80)
    print(f"{'Horizon':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 40)

    for horizon_name, metrics in horizon_results.items():
        print(f"{horizon_name:<10} {metrics['MAE']:<10.4f} "
              f"{metrics['RMSE']:<10.4f} {metrics['MAPE']:<10.2f}%")

    best_horizon = min(horizon_results, key=lambda h: horizon_results[h]['MAE'])
    print("=" * 80)
    print(f"BEST PERFORMANCE: {best_horizon} "
          f"(MAE: {horizon_results[best_horizon]['MAE']:.4f})")
    print("=" * 80 + "\n")


def run_experiment(config_path: str, override_config: Dict = None):
    config = load_config(config_path)

    if override_config:
        for key, value in override_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value

    if not validate_config(config):
        raise ValueError("Invalid configuration")

    # Resolve minibatch_size: command-line override wins, then config, then device default
    if 'minibatch_size' not in config.get('training', {}):
        config['training']['minibatch_size'] = _default_minibatch_size()
    minibatch_size = int(config['training']['minibatch_size'])

    set_seed(config['experiment']['seed'])
    device = (get_device() if config['experiment']['device'] == 'auto'
              else torch.device(config['experiment']['device']))

    experiment_name, logger, model_saver, visualizer = setup_experiment(config)

    print(f"Experiment:     {experiment_name}")
    print(f"Device:         {device}")
    print(f"Dataset:        {config['training']['test_dataset']}")
    print(f"Model:          MCPST-FSL")
    print(f"Minibatch size: {minibatch_size}")
    print("=" * 60)
    logger.log(f"Starting experiment: {experiment_name}")
    logger.log(f"Minibatch size: {minibatch_size}")

    data_manager = TrafficDataManager(config['data'], config['task'])
    dataloaders = data_manager.create_dataloaders(
        test_data=config['training']['test_dataset'],
        target_days=config['training']['target_days'],
        add_physics_features=config['training']['add_physics_features'],
        minibatch_size=minibatch_size,
    )

    model = create_model(config, device)
    print_model_summary(
        model,
        (
            config['task']['batch_size'],
            config['task']['his_num'],
            config['data'][config['training']['test_dataset']]['node_num'],
            config['model']['input_dim'],
        ),
    )

    trainer = create_trainer(model, config, device, logger)
    save_config(config, os.path.join(
        config['experiment']['save_dir'], f'{experiment_name}_config.yaml'
    ))

    with Timer() as training_timer:
        if config['few_shot']['enabled']:
            training_history = trainer.train(data_manager)
        else:
            training_history = trainer.train(dataloaders)

    training_time = training_timer.elapsed()
    print(f"Training completed in {format_time(training_time)}")
    logger.log(f"Training completed in {format_time(training_time)}")

    if config['logging']['save_model_checkpoints']:
        optimizer = (trainer.optimizer if hasattr(trainer, 'optimizer')
                     else trainer.meta_optimizer)
        model_path = model_saver.save_model(model, optimizer, is_best=True)
        logger.log(f"Model saved to: {model_path}")

    horizon_results, predictions_array, targets_array = evaluate_horizons_fixed(
        model, dataloaders['test'], device, config, data_manager, logger,
        minibatch_size=minibatch_size,
    )

    print_horizon_results(horizon_results, config['training']['test_dataset'], logger)

    if config['logging']['plot_predictions']:
        visualizer.plot_flow_predictions(
            predictions_array, targets_array,
            node_indices=config['evaluation']['plot_nodes'][:3],
            time_indices=config['evaluation']['plot_time_steps'][:3],
            save_name='final_predictions_scatter',
        )
        visualizer.plot_time_series(
            predictions_array, targets_array,
            node_indices=config['evaluation']['plot_nodes'][:2],
            save_name='final_timeseries',
        )
        visualizer.plot_horizon_performance(horizon_results, 'final_horizon_performance')
        visualizer.plot_error_distribution(predictions_array, targets_array, 'final_error_distribution')
        visualizer.plot_temporal_error_patterns(predictions_array, targets_array, 'final_temporal_patterns')

    final_horizon = max(horizon_results, key=lambda h: len(h))
    overall_results = horizon_results[final_horizon]

    print("OVERALL PERFORMANCE SUMMARY:")
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in overall_results:
            print(f"{metric:15}: {overall_results[metric]:.6f}")

    return {
        'experiment_name': experiment_name,
        'training_time': training_time,
        'horizon_results': horizon_results,
        'final_metrics': overall_results,
        'training_history': training_history,
        'config': config,
    }


# def main():
#     parser = argparse.ArgumentParser(
#         description='MCPST-FSL Traffic Flow Prediction',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     # ── Experiment ────────────────────────────────────────────────────────────
#     parser.add_argument('--config', type=str, default='config.yaml',help='Path to YAML config file')
#     parser.add_argument('--dataset', type=str, default=None, help='Override training.test_dataset')
#     parser.add_argument('--seed', type=int, default=None, help='Override experiment.seed')
#     parser.add_argument('--device', type=str, default=None, help='Override experiment.device (auto | cpu | cuda)')
#     parser.add_argument('--source_epochs', type=int, default=None, help='Override training.source_epochs')
#     parser.add_argument('--target_epochs', type=int, default=None, help='Override training.target_epochs')
#     parser.add_argument('--target_days', type=int, default=None, help='Few-shot target days (training.target_days)')
#     parser.add_argument('--source_lr', type=float, default=None, help='Source-domain learning rate (training.source_lr)')
#     parser.add_argument('--target_lr', type=float, default=None, help='Target-domain fine-tune LR (training.target_lr)')
#     parser.add_argument('--weight_decay', type=float, default=None, help='AdamW weight decay (training.weight_decay)')
#     parser.add_argument('--gradient_clip', type=float, default=None, help='Gradient clip norm (training.gradient_clip)')

#     # ── Minibatch / compute ───────────────────────────────────────────────────
#     _mb_default = _default_minibatch_size()
#     parser.add_argument(
#         '--minibatch_size', type=int, default=_mb_default,
#         help=(
#             'Samples per forward/backward pass. '
#             f'Defaults to {_mb_default} on this machine '
#             '(2048 GPU / 4096 CPU). Gradient accumulation ensures the '
#             'effective update equals processing the full batch at once.'
#         ),
#     )

#     # ── Model architecture ────────────────────────────────────────────────────
#     parser.add_argument('--in_channels', '--input_dim', dest='input_dim', type=int, default=None,
#                         help='Input feature dimension (model.input_dim). '
#                              'Auto-set to 6 (physics) or 2 (raw) unless overridden.')
#     parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden / embedding dimension (model.hidden_dim)')
#     parser.add_argument('--pred_len', type=int, default=None, help='Prediction horizon in steps (model.pred_len)')
#     parser.add_argument('--spatial_dim', type=int, default=None, help='Spatial embedding dimension (model.spatial_dim)')
#     parser.add_argument('--num_diffusion_steps', type=int, default=None, help='Heat-diffusion integration steps (model.num_diffusion_steps)')
#     parser.add_argument('--num_oscillator_steps', type=int, default=None, help='Kuramoto oscillator steps (model.num_oscillator_steps)')
#     parser.add_argument('--num_eigen_vectors', type=int, default=None, help='Spectral eigenvectors used (model.num_eigen_vectors)')
#     parser.add_argument('--dropout', type=float, default=None, help='Dropout rate (model.dropout)')
#     parser.add_argument('--node_minibatch_cpu', type=int, default=None,
#                         help='Nodes per temporal-encoder call on CPU (model.node_minibatch_cpu)')
#     parser.add_argument('--node_minibatch_gpu', type=int, default=None,
#                         help='Nodes per temporal-encoder call on GPU (model.node_minibatch_gpu)')

#     # ── Physics loss weights ──────────────────────────────────────────────────
#     # These directly control which loss components are non-zero in the output.
#     # Setting alpha_flow=0 is the most common cause of "Flow: 0.0000" — ensure
#     # it stays > 0 (config default is 2.0).
#     parser.add_argument('--alpha_flow', type=float, default=None,
#                         help='Flow prediction loss weight (model.physics.alpha_flow). '
#                              'Must be > 0 or flow_loss will be 0 in logs.')
#     parser.add_argument('--alpha_spatial', type=float, default=None,
#                         help='Spatial target loss weight (model.physics.alpha_spatial)')
#     parser.add_argument('--alpha_temporal', type=float, default=None,
#                         help='Temporal target loss weight (model.physics.alpha_temporal)')
#     parser.add_argument('--alpha_physics', type=float, default=None,
#                         help='Physics regularisation weight (model.physics.alpha_physics)')
#     parser.add_argument('--alpha_consistency', type=float, default=None,
#                         help='Physics-weight consistency loss (model.physics.alpha_consistency)')
#     parser.add_argument('--alpha_uncertainty', type=float, default=None,
#                         help='Uncertainty calibration loss weight '
#                              '(model.physics.alpha_uncertainty). '
#                              'Must be > 0 or uncertainty_loss will be 0 in logs.')
#     parser.add_argument('--alpha_coupling', type=float, default=None,
#                         help='Coupling regularisation weight (model.physics.alpha_coupling)')

#     # ── Few-shot ──────────────────────────────────────────────────────────────
#     parser.add_argument('--few_shot', action='store_true', default=None,
#                         help='Enable few-shot learning (few_shot.enabled)')
#     parser.add_argument('--support_size', type=int, default=None,
#                         help='Support set size per task (few_shot.support_size)')
#     parser.add_argument('--query_size', type=int, default=None,
#                         help='Query set size per task (few_shot.query_size)')
#     parser.add_argument('--num_tasks', type=int, default=None,
#                         help='Tasks per meta-update (few_shot.num_tasks)')
#     parser.add_argument('--adaptation_steps', type=int, default=None,
#                         help='Inner-loop gradient steps (few_shot.adaptation_steps)')
#     parser.add_argument('--inner_lr', type=float, default=None,
#                         help='Inner-loop SGD learning rate (few_shot.inner_lr)')
#     parser.add_argument('--outer_lr', type=float, default=None,
#                         help='Outer-loop (meta) learning rate (few_shot.outer_lr)')
#     parser.add_argument('--gradient_clip_inner', type=float, default=None,
#                         help='Grad-clip norm for inner loop (few_shot.gradient_clip_inner)')
#     parser.add_argument('--gradient_clip_outer', type=float, default=None,
#                         help='Grad-clip norm for outer loop (few_shot.gradient_clip_outer)')

#     args = parser.parse_args()
def main():
    parser = argparse.ArgumentParser(
        description='MCPST-FSL Traffic Flow Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, default='metr-la', help='Override training.test_dataset')
    parser.add_argument('--seed', type=int, default=42, help='Override experiment.seed')
    parser.add_argument('--device', type=str, default='auto', help='Override experiment.device (auto | cpu | cuda)')
    parser.add_argument('--source_epochs', type=int, default=250, help='Override training.source_epochs')
    parser.add_argument('--target_epochs', type=int, default=250, help='Override training.target_epochs')
    parser.add_argument('--target_days', type=int, default=3, help='Few-shot target days (training.target_days)')
    parser.add_argument('--source_lr', type=float, default=0.0003, help='Source-domain learning rate (training.source_lr)')
    parser.add_argument('--target_lr', type=float, default=0.0001, help='Target-domain fine-tune LR (training.target_lr)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='AdamW weight decay (training.weight_decay)')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clip norm (training.gradient_clip)')

    _mb_default = _default_minibatch_size()
    parser.add_argument(
        '--minibatch_size', type=int, default=_mb_default,
        help=f'Samples per forward/backward pass. Defaults to {_mb_default} on this machine (2048 GPU / 4096 CPU).'
    )

    parser.add_argument('--input_dim', type=int, default=6, help='Input feature dimension (model.input_dim)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden / embedding dimension (model.hidden_dim)')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction horizon in steps (model.pred_len)')
    parser.add_argument('--spatial_dim', type=int, default=2, help='Spatial embedding dimension (model.spatial_dim)')
    parser.add_argument('--num_diffusion_steps', type=int, default=6, help='Heat-diffusion integration steps (model.num_diffusion_steps)')
    parser.add_argument('--num_oscillator_steps', type=int, default=10, help='Kuramoto oscillator steps (model.num_oscillator_steps)')
    parser.add_argument('--num_eigen_vectors', type=int, default=8, help='Spectral eigenvectors used (model.num_eigen_vectors)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (model.dropout)')
    parser.add_argument('--node_minibatch_cpu', type=int, default=4096, help='Nodes per temporal-encoder call on CPU (model.node_minibatch_cpu)')
    parser.add_argument('--node_minibatch_gpu', type=int, default=1024, help='Nodes per temporal-encoder call on GPU (model.node_minibatch_gpu)')

    parser.add_argument('--alpha_flow', type=float, default=2.0, help='Flow prediction loss weight (model.physics.alpha_flow)')
    parser.add_argument('--alpha_spatial', type=float, default=0.08, help='Spatial target loss weight (model.physics.alpha_spatial)')
    parser.add_argument('--alpha_temporal', type=float, default=0.08, help='Temporal target loss weight (model.physics.alpha_temporal)')
    parser.add_argument('--alpha_physics', type=float, default=0.03, help='Physics regularisation weight (model.physics.alpha_physics)')
    parser.add_argument('--alpha_consistency', type=float, default=0.015, help='Physics-weight consistency loss (model.physics.alpha_consistency)')
    parser.add_argument('--alpha_uncertainty', type=float, default=0.01, help='Uncertainty calibration loss weight (model.physics.alpha_uncertainty)')
    parser.add_argument('--alpha_coupling', type=float, default=0.05, help='Coupling regularisation weight (model.physics.alpha_coupling)')

    parser.add_argument('--few_shot', action='store_true', default=True, help='Enable few-shot learning (few_shot.enabled)')
    parser.add_argument('--support_size', type=int, default=5, help='Support set size per task (few_shot.support_size)')
    parser.add_argument('--query_size', type=int, default=10, help='Query set size per task (few_shot.query_size)')
    parser.add_argument('--num_tasks', type=int, default=1, help='Tasks per meta-update (few_shot.num_tasks)')
    parser.add_argument('--adaptation_steps', type=int, default=3, help='Inner-loop gradient steps (few_shot.adaptation_steps)')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner-loop SGD learning rate (few_shot.inner_lr)')
    parser.add_argument('--outer_lr', type=float, default=0.0005, help='Outer-loop (meta) learning rate (few_shot.outer_lr)')
    parser.add_argument('--gradient_clip_inner', type=float, default=1.0, help='Grad-clip norm for inner loop (few_shot.gradient_clip_inner)')
    parser.add_argument('--gradient_clip_outer', type=float, default=1.0, help='Grad-clip norm for outer loop (few_shot.gradient_clip_outer)')
    parser.add_argument('--meta_batch_size', type=int, default=1, help='Number of tasks per meta-batch (use 1 for chaos preservation)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help='Accumulate gradients over N steps')

    args = parser.parse_args()
    # ── Build override dict ───────────────────────────────────────────────────
    # Only keys that were explicitly supplied on the command line are written
    # so that config-file defaults are never silently overwritten.

    override_config: Dict = {}

    def _set(section: str, key: str, value):
        """Write value into override_config[section][key] if value is not None."""
        if value is not None:
            override_config.setdefault(section, {})[key] = value

    # Experiment
    _set('training',   'test_dataset', args.dataset)
    _set('experiment', 'seed',         args.seed)
    _set('experiment', 'device',       args.device)

    # Training schedule
    _set('training', 'source_epochs', args.source_epochs)
    _set('training', 'target_epochs', args.target_epochs)
    _set('training', 'target_days',   args.target_days)
    _set('training', 'source_lr',     args.source_lr)
    _set('training', 'target_lr',     args.target_lr)
    _set('training', 'weight_decay',  args.weight_decay)
    _set('training', 'gradient_clip', args.gradient_clip)

    # Minibatch (always written — has a computed default, not None)
    override_config.setdefault('training', {})['minibatch_size'] = args.minibatch_size

    # Model architecture
    _set('model', 'input_dim',            args.input_dim)
    _set('model', 'hidden_dim',           args.hidden_dim)
    _set('model', 'pred_len',             args.pred_len)
    _set('model', 'spatial_dim',          args.spatial_dim)
    _set('model', 'num_diffusion_steps',  args.num_diffusion_steps)
    _set('model', 'num_oscillator_steps', args.num_oscillator_steps)
    _set('model', 'num_eigen_vectors',    args.num_eigen_vectors)
    _set('model', 'dropout',              args.dropout)
    _set('model', 'node_minibatch_cpu',   args.node_minibatch_cpu)
    _set('model', 'node_minibatch_gpu',   args.node_minibatch_gpu)

    # Physics loss weights — written into model.physics sub-dict
    # Any weight left at None keeps the config-file value (default alpha_flow=2.0,
    # alpha_uncertainty=0.01).  Setting either to 0 from the CLI will suppress
    # that component and produce 0.0 in training logs — intentionally visible.
    physics_args = {
        'alpha_flow':        args.alpha_flow,
        'alpha_spatial':     args.alpha_spatial,
        'alpha_temporal':    args.alpha_temporal,
        'alpha_physics':     args.alpha_physics,
        'alpha_consistency': args.alpha_consistency,
        'alpha_uncertainty': args.alpha_uncertainty,
        'alpha_coupling':    args.alpha_coupling,
    }
    for k, v in physics_args.items():
        if v is not None:
            override_config.setdefault('model', {}).setdefault('physics', {})[k] = v

    # Few-shot
    if args.few_shot:
        override_config.setdefault('few_shot', {})['enabled'] = True
    _set('few_shot', 'support_size',         args.support_size)
    _set('few_shot', 'query_size',           args.query_size)
    _set('few_shot', 'num_tasks',            args.num_tasks)
    _set('few_shot', 'adaptation_steps',     args.adaptation_steps)
    _set('few_shot', 'inner_lr',             args.inner_lr)
    _set('few_shot', 'outer_lr',             args.outer_lr)
    _set('few_shot', 'gradient_clip_inner',  args.gradient_clip_inner)
    _set('few_shot', 'gradient_clip_outer',  args.gradient_clip_outer)

    results = run_experiment(args.config, override_config or None)

    print("\nExperiment completed.")
    horizon_results = results['horizon_results']
    best_horizon = min(horizon_results, key=lambda h: horizon_results[h]['MAE'])
    best_metrics = horizon_results[best_horizon]

    print(f"Best Horizon: {best_horizon}")
    print(f"Best MAE:     {best_metrics['MAE']:.6f}")
    print(f"Best RMSE:    {best_metrics['RMSE']:.6f}")
    print(f"Best MAPE:    {best_metrics['MAPE']:.2f}%")

    return results


if __name__ == "__main__":
    results = main()