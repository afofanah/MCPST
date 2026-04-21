import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Optional
import pickle


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def robust_normalize_adjacency(A):
    A = A.astype(np.float32)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    D = np.maximum(np.array(np.sum(A, axis=1), dtype=np.float32).reshape((-1,)), 1e-5)
    diag = np.reciprocal(np.sqrt(D)).astype(np.float32)
    return np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1))).astype(np.float32)


def enhanced_physics_features(data, adjacency_matrix):
    num_timesteps, num_nodes, original_features = data.shape

    if adjacency_matrix.shape[0] != num_nodes:
        min_nodes = min(adjacency_matrix.shape[0], num_nodes)
        adjacency_matrix = adjacency_matrix[:min_nodes, :min_nodes]
        data = data[:, :min_nodes, :]
        num_nodes = min_nodes

    physics_features = np.zeros((num_timesteps, num_nodes, 4), dtype=np.float32)
    degree = np.sum(adjacency_matrix, axis=1)
    degree_normalized = degree / (np.max(degree) + 1e-8)

    if original_features > 0:
        flow_data = data[:, :, 0]
        physics_features[:, :, 0] = np.broadcast_to(
            degree_normalized[np.newaxis, :], (num_timesteps, num_nodes)
        )

        if num_timesteps > 2:
            flow_variance = np.var(flow_data, axis=0)
            physics_features[:, :, 1] = np.broadcast_to(
                (flow_variance / (np.max(flow_variance) + 1e-8))[np.newaxis, :],
                (num_timesteps, num_nodes)
            )
            temporal_gradient = np.gradient(flow_data, axis=0)
            temporal_std = np.std(temporal_gradient) + 1e-6
            physics_features[:, :, 3] = np.clip(temporal_gradient / temporal_std, -3, 3)
        else:
            physics_features[:, :, 1] = 0.1
            physics_features[:, :, 3] = 0.0

        neighbor_influence = np.zeros((num_timesteps, num_nodes))
        for node in range(num_nodes):
            neighbors = np.where(adjacency_matrix[node, :] > 0)[0]
            if len(neighbors) > 0:
                neighbor_influence[:, node] = np.mean(flow_data[:, neighbors], axis=1)
            else:
                neighbor_influence[:, node] = flow_data[:, node]

        neighbor_mean = np.mean(neighbor_influence)
        neighbor_std = np.std(neighbor_influence) + 1e-6
        physics_features[:, :, 2] = np.clip(
            (neighbor_influence - neighbor_mean) / neighbor_std, -3, 3
        )
    else:
        physics_features[:, :, 0] = np.broadcast_to(
            degree_normalized[np.newaxis, :], (num_timesteps, num_nodes)
        )
        physics_features[:, :, 1:] = 0.0

    return np.concatenate([data, physics_features], axis=2).astype(np.float32)


def create_spatiotemporal_targets(flow_data, adjacency_matrix):
    num_nodes, time_len = flow_data.shape
    spatial_target = np.zeros((num_nodes, time_len, 2), dtype=np.float32)
    temporal_target = np.zeros((num_nodes, time_len), dtype=np.float32)

    eigenvals, eigenvecs = np.linalg.eigh(adjacency_matrix + 1e-6 * np.eye(num_nodes))

    for node in range(num_nodes):
        neighbors = np.where(adjacency_matrix[node, :] > 0)[0]
        node_flow = flow_data[node, :]
        spatial_embedding = (eigenvecs[node, :2] if eigenvecs.shape[1] >= 2
                             else np.array([node % 10, node // 10], dtype=np.float32))

        for t in range(time_len):
            flow_magnitude = np.abs(node_flow[t]) + 1e-6
            time_factor = 2 * np.pi * t / max(time_len, 1)
            spatial_target[node, t, 0] = spatial_embedding[0] + 0.1 * flow_magnitude * np.cos(time_factor)
            spatial_target[node, t, 1] = spatial_embedding[1] + 0.1 * flow_magnitude * np.sin(time_factor)

        if len(neighbors) > 0 and time_len > 1:
            neighbor_mean = np.mean(flow_data[neighbors, :], axis=0)
            correlation = np.corrcoef(node_flow, neighbor_mean)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            temporal_target[node, :] = correlation * np.gradient(node_flow)
        else:
            temporal_target[node, :] = (np.gradient(node_flow) if time_len > 1
                                        else np.zeros(time_len))

    spatial_mean = np.mean(spatial_target, axis=(0, 1), keepdims=True)
    spatial_std = np.std(spatial_target, axis=(0, 1), keepdims=True) + 1e-6
    spatial_target = (spatial_target - spatial_mean) / spatial_std

    temporal_mean = np.mean(temporal_target)
    temporal_std = np.std(temporal_target) + 1e-6
    temporal_target = (temporal_target - temporal_mean) / temporal_std

    return spatial_target, temporal_target


def enhanced_sequences(X, his_len, pred_len, adjacency_matrix=None, include_targets=True):
    total_len = his_len + pred_len
    num_sequences = max(1, X.shape[0] - total_len + 1)
    step_size = max(1, num_sequences // 500) if num_sequences > 500 else 1
    indices = list(range(0, num_sequences, step_size))

    features = []
    targets = {'flow': []}
    if include_targets:
        targets['spatial'] = []
        targets['temporal'] = []

    for i in indices:
        feature_seq = X[i:i + his_len]
        # X shape: (timesteps, num_nodes, features).
        # Raw slice is (pred_len, num_nodes); transpose to (num_nodes, pred_len)
        # so the stacked tensor is (N, num_nodes, pred_len), matching the model
        # output shape (batch, num_nodes, pred_len).
        flow_target = X[i + his_len:i + total_len, :, 0].T  # (num_nodes, pred_len)
        if np.any(np.isnan(feature_seq)) or np.any(np.isnan(flow_target)):
            continue
        features.append(feature_seq)
        targets['flow'].append(flow_target)
        if include_targets and adjacency_matrix is not None:
            # flow_target is already (num_nodes, pred_len) — pass directly
            spatial_target, temporal_target = create_spatiotemporal_targets(
                flow_target, adjacency_matrix
            )
            targets['spatial'].append(spatial_target)
            targets['temporal'].append(temporal_target)

    if not features:
        dummy_feature = np.zeros((1, his_len, X.shape[1], X.shape[2]), dtype=np.float32)
        # flow dummy: (1, num_nodes, pred_len) — consistent with transposed convention
        dummy_flow = np.zeros((1, X.shape[1], pred_len), dtype=np.float32)
        return torch.from_numpy(dummy_feature), {'flow': torch.from_numpy(dummy_flow)}

    features_tensor = torch.from_numpy(np.array(features, dtype=np.float32))
    target_tensors = {
        k: torch.from_numpy(np.array(v, dtype=np.float32))
        for k, v in targets.items() if v
    }
    return features_tensor, target_tensors


class TrafficDataset(Dataset):
    def __init__(self, data_args, task_args, stage='source', test_data='metr-la',
                 add_target=True, target_days=3, add_physics_features=True,
                 minibatch_size: int = 4096):
        super().__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.test_data = test_data
        self.target_days = target_days
        self.add_physics_features = add_physics_features

        # ------------------------------------------------------------------
        # minibatch_size: controls how many samples __getitem__ draws for the
        # source stage per call (one DataLoader step = one gradient update).
        # Larger values → richer gradient estimates per step; defaults differ
        # by device (4096 CPU / 2048 GPU) to balance memory and compute.
        # __len__ is scaled inversely so the total data volume per epoch stays
        # constant (legacy was 500 steps × task_args['batch_size'] samples).
        # ------------------------------------------------------------------
        self.minibatch_size = minibatch_size

        self.cache_dir = 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)

        self.A_list = {}
        self.edge_index_list = {}
        self.edge_attr_list = {}
        self.x_list = {}
        self.y_list = {}
        self.means_list = {}
        self.stds_list = {}

        self.load_all_data(stage, test_data, add_target)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def get_cache_key(self, dataset_name, stage):
        # v3: flow_target transposed to (num_nodes, pred_len) to match model output shape
        return f"mcpst_{dataset_name}_{stage}_{self.target_days}_{self.add_physics_features}_v3"

    def load_from_cache(self, cache_key):
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_to_cache(self, cache_key, data):
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_all_data(self, stage, test_data, add_target):
        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'source':
            self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
            if add_target:
                self.data_list = np.append(self.data_list, test_data)
        else:
            self.data_list = np.array([test_data])
        for dataset_name in self.data_list:
            self.load_dataset(dataset_name, stage)

    def load_dataset(self, dataset_name, stage):
        cache_key = self.get_cache_key(dataset_name, stage)
        cached = self.load_from_cache(cache_key)
        if cached:
            self.A_list[dataset_name] = cached['A']
            self.edge_index_list[dataset_name] = cached['edge_index']
            self.edge_attr_list[dataset_name] = cached['edge_attr']
            self.x_list[dataset_name] = cached['x']
            self.y_list[dataset_name] = cached['y']
            self.means_list[dataset_name] = cached['means']
            self.stds_list[dataset_name] = cached['stds']
            return

        config = self.data_args[dataset_name]
        A_raw = np.load(config['adjacency_matrix_path'])
        A_normalized = robust_normalize_adjacency(A_raw)
        self.A_list[dataset_name] = torch.from_numpy(A_normalized).float()

        edge_index, edge_attr = self.create_edge_features(A_raw)
        self.edge_index_list[dataset_name] = edge_index
        self.edge_attr_list[dataset_name] = edge_attr

        X_raw = np.load(config['dataset_path']).astype(np.float32)
        if X_raw.ndim == 3:
            if X_raw.shape[2] > X_raw.shape[0] and X_raw.shape[1] < 1000:
                X_raw = X_raw.transpose((2, 0, 1))
            elif X_raw.shape[1] > X_raw.shape[0] and X_raw.shape[1] > X_raw.shape[2]:
                X_raw = X_raw.transpose((1, 2, 0))

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=100.0, neginf=0.0)
        global_mean = np.mean(X_raw)
        global_std = max(np.std(X_raw), 1e-6)
        X_normalized = np.clip((X_raw - global_mean) / global_std, -5, 5)

        if self.add_physics_features:
            X_normalized = enhanced_physics_features(X_normalized, A_raw)

        if stage == 'target':
            X_normalized = X_normalized[:min(288 * self.target_days, X_normalized.shape[0])]
        elif stage == 'test':
            split_idx = int(X_normalized.shape[0] * 0.8)
            X_normalized = X_normalized[split_idx:]

        x_data, y_data = enhanced_sequences(
            X_normalized, self.his_num, self.pred_num, A_raw, include_targets=True
        )
        self.x_list[dataset_name] = x_data
        self.y_list[dataset_name] = y_data
        self.means_list[dataset_name] = global_mean
        self.stds_list[dataset_name] = global_std

        self.save_to_cache(cache_key, {
            'A': self.A_list[dataset_name],
            'edge_index': self.edge_index_list[dataset_name],
            'edge_attr': self.edge_attr_list[dataset_name],
            'x': self.x_list[dataset_name],
            'y': self.y_list[dataset_name],
            'means': self.means_list[dataset_name],
            'stds': self.stds_list[dataset_name],
        })

    def create_edge_features(self, adjacency_matrix):
        rows, cols = np.where(adjacency_matrix > 0)
        edge_weights = adjacency_matrix[rows, cols]
        return (
            torch.tensor(np.vstack([rows, cols]), dtype=torch.long),
            torch.tensor(edge_weights, dtype=torch.float),
        )

    # ------------------------------------------------------------------
    # Few-shot helpers
    # ------------------------------------------------------------------

    def create_few_shot_batch(self, x_data, y_data, support_size=8, query_size=12):
        total_samples = x_data.shape[0]
        max_samples = min(total_samples, support_size + query_size)
        actual_support = min(support_size, max_samples // 2)
        actual_query = max_samples - actual_support
        indices = torch.randperm(total_samples)[:max_samples]
        support_idx = indices[:actual_support]
        query_idx = indices[actual_support:actual_support + actual_query]

        support_x = x_data[support_idx]
        query_x = x_data[query_idx]

        if isinstance(y_data, dict):
            support_y = {k: v[support_idx] for k, v in y_data.items()}
            query_y = {k: v[query_idx] for k, v in y_data.items()}
        else:
            support_y = y_data[support_idx]
            query_y = y_data[query_idx]

        return support_x, support_y, query_x, query_y

    def get_few_shot_tasks(self, num_tasks, support_size=8, query_size=12):
        support_tasks, query_tasks = [], []
        support_adjacencies, query_adjacencies = [], []
        dataset_name = random.choice(self.data_list)

        for _ in range(num_tasks):
            x_data = self.x_list[dataset_name]
            y_data = self.y_list[dataset_name]
            support_x, support_y, query_x, query_y = self.create_few_shot_batch(
                x_data, y_data, support_size, query_size
            )
            node_num = self.A_list[dataset_name].shape[0]
            adjacency = self.A_list[dataset_name]

            support_data = Data(
                node_num=node_num, x=support_x, y=support_y,
                data_name=dataset_name
            )
            support_data.edge_index = self.edge_index_list[dataset_name]

            query_data = Data(
                node_num=node_num, x=query_x, y=query_y,
                data_name=dataset_name
            )
            query_data.edge_index = self.edge_index_list[dataset_name]

            support_tasks.append(support_data)
            query_tasks.append(query_data)
            support_adjacencies.append(adjacency)
            query_adjacencies.append(adjacency)

        return support_tasks, support_adjacencies, query_tasks, query_adjacencies

    # ------------------------------------------------------------------
    # Core dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        if self.stage == 'source':
            # ------------------------------------------------------------------
            # Draw self.minibatch_size samples per step instead of the legacy
            # task_args['batch_size'] (typically 8).  Larger batches give
            # better gradient estimates; __len__ is scaled down proportionally
            # so total data seen per epoch stays the same.
            # ------------------------------------------------------------------
            dataset_name = random.choice(self.data_list)
            total_samples = self.x_list[dataset_name].shape[0]
            sample_size = min(self.minibatch_size, total_samples)

            indices = torch.randperm(total_samples)[:sample_size]
            x_data = self.x_list[dataset_name][indices]
            y_data = {k: v[indices] for k, v in self.y_list[dataset_name].items()}
        else:
            dataset_name = self.data_list[0]
            index = index % self.x_list[dataset_name].shape[0]
            x_data = self.x_list[dataset_name][index:index + 1]
            y_data = {k: v[index:index + 1] for k, v in self.y_list[dataset_name].items()}

        node_num = self.A_list[dataset_name].shape[0]
        data = Data(node_num=node_num, x=x_data, y=y_data, data_name=dataset_name)
        data.edge_index = self.edge_index_list[dataset_name]
        return data, self.A_list[dataset_name]

    def __len__(self):
        if self.stage == 'source':
            # Keep total data volume constant:
            #   legacy = 500 steps × task_args['batch_size'] samples per step
            # With minibatch_size samples per step, we need fewer steps.
            base_steps = 500
            base_batch = int(self.task_args.get('batch_size', 8))
            return max(1, base_steps * base_batch // self.minibatch_size)
        return self.x_list[self.data_list[0]].shape[0]


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class TrafficDataManager:
    def __init__(self, data_args, task_args):
        self.data_args = data_args
        self.task_args = task_args
        self.datasets = {}

    def create_dataset(self, stage, test_data, minibatch_size: int = 4096, **kwargs):
        dataset = TrafficDataset(
            data_args=self.data_args,
            task_args=self.task_args,
            stage=stage,
            test_data=test_data,
            minibatch_size=minibatch_size,
            **kwargs,
        )
        self.datasets[f"{stage}_{test_data}"] = dataset
        return dataset

    def create_dataloaders(self, test_data, target_days=3,
                           add_physics_features=True,
                           minibatch_size: int = 4096):
        """
        Build DataLoaders for all pipeline stages.

        ``minibatch_size`` is forwarded to every TrafficDataset so that:
          • Source __getitem__ draws ``minibatch_size`` samples per call
            (the DataLoader itself keeps batch_size=1 because the dataset
            already returns a full pre-batched item).
          • Test DataLoader uses ``minibatch_size`` as its batch_size so
            inference runs in large, efficient chunks instead of tiny batches.
        """
        dataloaders = {}

        # ---- source: pre-batched inside __getitem__, DataLoader wraps it ---
        source_dataset = self.create_dataset(
            'source', test_data,
            add_physics_features=add_physics_features,
            minibatch_size=minibatch_size,
        )
        dataloaders['source'] = DataLoader(
            source_dataset, batch_size=1, shuffle=True,
            num_workers=0, pin_memory=False,
        )

        # ---- target (few-shot fine-tune / validation data) ------------------
        target_dataset = self.create_dataset(
            'target', test_data,
            target_days=target_days,
            add_physics_features=add_physics_features,
            minibatch_size=minibatch_size,
        )
        dataloaders['target'] = DataLoader(
            target_dataset, batch_size=1, shuffle=True,
            num_workers=0, pin_memory=False,
        )

        # ---- test: larger batch_size for efficient inference ----------------
        test_dataset = self.create_dataset(
            'test', test_data,
            add_physics_features=add_physics_features,
            minibatch_size=minibatch_size,
        )
        test_batch = min(minibatch_size, len(test_dataset))
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=max(1, test_batch),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        return dataloaders

    def get_statistics(self):
        stats = {}
        for key, dataset in self.datasets.items():
            dataset_name = dataset.data_list[0]
            stats[key] = {
                'name': dataset_name,
                'samples': len(dataset),
                'node_count': dataset.A_list[dataset_name].shape[0],
                'feature_dim': dataset.x_list[dataset_name].shape[-1],
                'physics_features': dataset.add_physics_features,
                'mean': dataset.means_list[dataset_name],
                'std': dataset.stds_list[dataset_name],
            }
        return stats