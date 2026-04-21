import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_scales=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        self.short_term = nn.LSTM(input_dim, hidden_dim // 8, batch_first=True, dropout=dropout, num_layers=2)
        self.medium_term = nn.LSTM(input_dim, hidden_dim // 8, batch_first=True, dropout=dropout, num_layers=2)
        self.long_term = nn.LSTM(input_dim, hidden_dim // 8, batch_first=True, dropout=dropout, num_layers=2)
        self.very_long_term = nn.LSTM(input_dim, hidden_dim // 8, batch_first=True, dropout=dropout, num_layers=2)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)

        lstm_concat_dim = 4 * (hidden_dim // 8)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, lstm_concat_dim) * 0.1)
        self.lstm_projection = nn.Linear(lstm_concat_dim, hidden_dim)

        self.seasonal_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.LayerNorm(hidden_dim // 8)
        )

        self.trend_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 8, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1)
        )

        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        short_out, _ = self.short_term(x)

        if seq_len > 2:
            medium_out, _ = self.medium_term(x[:, ::2, :])
            if medium_out.size(1) != seq_len:
                medium_out = F.interpolate(medium_out.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        else:
            medium_out = short_out

        if seq_len > 4:
            long_out, _ = self.long_term(x[:, ::4, :])
            if long_out.size(1) != seq_len:
                long_out = F.interpolate(long_out.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        else:
            long_out = short_out

        if seq_len > 8:
            very_long_out, _ = self.very_long_term(x[:, ::8, :])
            if very_long_out.size(1) != seq_len:
                very_long_out = F.interpolate(very_long_out.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        else:
            very_long_out = short_out

        combined_lstm = torch.cat([short_out, medium_out, long_out, very_long_out], dim=-1)
        combined_lstm = combined_lstm + self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        combined_lstm = self.lstm_projection(combined_lstm)

        transformer_out = self.transformer_encoder(combined_lstm)
        memory_attended, _ = self.memory_attention(transformer_out, transformer_out, transformer_out)

        seasonal_features = self.seasonal_detector(x)
        seasonal_pooled = self.adaptive_pool(seasonal_features.transpose(1, 2)).squeeze(-1)
        trend_features = self.trend_extractor(x.transpose(1, 2))
        trend_pooled = self.adaptive_pool(trend_features).squeeze(-1)

        memory_pooled = torch.mean(memory_attended, dim=1)
        transformer_pooled = torch.mean(transformer_out, dim=1)

        if seasonal_pooled.dim() == 1:
            seasonal_pooled = seasonal_pooled.unsqueeze(1)
        if trend_pooled.dim() == 1:
            trend_pooled = trend_pooled.unsqueeze(1)

        seasonal_expanded = seasonal_pooled.expand(-1, memory_pooled.size(1)) if seasonal_pooled.size(1) == 1 else seasonal_pooled
        trend_expanded = trend_pooled.expand(-1, memory_pooled.size(1)) if trend_pooled.size(1) == 1 else trend_pooled

        all_features = torch.cat([memory_pooled, transformer_pooled, seasonal_expanded, trend_expanded], dim=-1)
        output = self.fusion_layers(all_features)
        output = output + (transformer_pooled if transformer_pooled.size(-1) == output.size(-1) else memory_pooled)
        return self.norm(output)


class HorizonSpecificPredictor(nn.Module):
    def __init__(self, hidden_dim, num_nodes, prediction_horizon, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.prediction_horizon = prediction_horizon
        self.output_dim = output_dim

        self.short_term_head = self._create_horizon_head(hidden_dim, "short")
        self.medium_term_head = self._create_horizon_head(hidden_dim, "medium")
        self.long_term_head = self._create_horizon_head(hidden_dim, "long")

        self.horizon_uncertainty = nn.ModuleDict({
            'short': self._create_uncertainty_head(hidden_dim // 2),
            'medium': self._create_uncertainty_head(hidden_dim // 2),
            'long': self._create_uncertainty_head(hidden_dim // 2)
        })

        self.horizon_weights = nn.Parameter(torch.ones(prediction_horizon))
        # Static per-node aggregator: applied independently to each node so
        # the shape is (hidden_dim//2 → pred_len), node-count-agnostic.
        # Replaces the old _get_or_create_aggregator which registered new
        # nn.Linear modules at forward time (different node counts per dataset
        # created differently-shaped params, breaking deepcopy + zip matching).
        self.per_node_aggregator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, prediction_horizon),
        )

    def _create_horizon_head(self, hidden_dim, horizon_type):
        if horizon_type == "short":
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU()
            )
        elif horizon_type == "medium":
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Dropout(0.15), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU()
            )
        else:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                nn.Dropout(0.15), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU()
            )

    def _create_uncertainty_head(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.GELU(),
            nn.Linear(input_dim // 2, 1), nn.Softplus()
        )

    def forward(self, x):
        batch_size, num_nodes_actual = x.size(0), x.size(1)
        short_features = self.short_term_head(x)
        medium_features = self.medium_term_head(x)
        long_features = self.long_term_head(x)

        horizon_weights = F.softmax(self.horizon_weights, dim=0)
        combined_features = []
        uncertainties = []

        for h in range(self.prediction_horizon):
            if h < 2:
                features = short_features
                uncertainty = self.horizon_uncertainty['short'](features)
            elif h < 4:
                features = medium_features
                uncertainty = self.horizon_uncertainty['medium'](features)
            else:
                features = long_features
                uncertainty = self.horizon_uncertainty['long'](features)
            combined_features.append(features * horizon_weights[h])
            uncertainties.append(uncertainty)

        # Stack horizon features: (batch, num_nodes, hidden_dim//2)
        # Apply per-node aggregator → (batch, num_nodes, pred_len).
        # PyTorch broadcasts nn.Linear over the node dimension automatically.
        node_features_agg = torch.stack(combined_features, dim=1).mean(dim=1)
        prediction = self.per_node_aggregator(node_features_agg)
        # prediction: (batch, num_nodes, pred_len)

        # uncertainty: stack over horizons then squeeze
        # Each uncertainty: (batch, num_nodes, 1) → stack dim=2 → (batch, num_nodes, pred_len, 1)
        # squeeze(-1) → (batch, num_nodes, pred_len)
        uncertainty = torch.stack(uncertainties, dim=2).squeeze(-1)  # (batch, num_nodes, pred_len)
        return prediction, uncertainty


class StabilizingGCN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.linear_transform = nn.Linear(feature_dim, hidden_dim)
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
        self.output_transform = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = features.shape
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.linear_transform(features)
        x = torch.bmm(adjacency, x)
        x = F.relu(self.graph_conv(x))
        x = self.dropout(x)
        x = self.output_transform(x)
        return x + features


class ThermodynamicModule(nn.Module):
    def __init__(self, feature_dim: int, num_steps: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        self.thermal_conductivity = nn.Parameter(torch.tensor(0.1))
        self.heat_capacity = nn.Parameter(torch.tensor(1.0))
        self.heat_source_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.ReLU(),
            nn.Linear(feature_dim // 2, 1), nn.Sigmoid()
        )
        self.flow_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4), nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )

    def heat_diffusion(self, features: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = features.shape
        if num_nodes != laplacian.shape[-1]:
            min_size = min(num_nodes, laplacian.shape[-1])
            features = features[:, :min_size, :]
            laplacian = laplacian[:, :min_size, :min_size]

        heat_sources = self.heat_source_predictor(features)
        current_state = features * heat_sources
        conductivity = torch.clamp(self.thermal_conductivity, 0.01, 0.3)
        capacity = torch.clamp(self.heat_capacity, 0.5, 2.0)
        dt = 0.1 / self.num_steps

        for _ in range(self.num_steps):
            gradient = torch.bmm(laplacian, current_state)
            current_state = torch.clamp(current_state + dt * conductivity / capacity * gradient, -2.0, 2.0)
        return current_state

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        degree = torch.sum(adjacency, dim=-1, keepdim=True) + 1e-6
        laplacian = torch.diag_embed(degree.squeeze(-1)) - adjacency
        diffused_features = self.heat_diffusion(features, laplacian)
        return {
            'diffused_features': diffused_features,
            'flow_predictions': self.flow_head(diffused_features).squeeze(-1),
            'heat_sources': self.heat_source_predictor(features).squeeze(-1)
        }


class KuramotoModule(nn.Module):
    def __init__(self, feature_dim: int, num_steps: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        self.frequency_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
        self.coupling_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4), nn.ReLU(),
            nn.Linear(feature_dim // 4, 1), nn.Sigmoid()
        )
        self.phase_to_flow = nn.Sequential(
            nn.Linear(feature_dim + 2, feature_dim // 2), nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        self.global_coupling = nn.Parameter(torch.tensor(0.3))

    def kuramoto_dynamics(self, features: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frequencies = self.frequency_predictor(features).squeeze(-1)
        local_coupling = self.coupling_predictor(features).squeeze(-1)
        phases = torch.atan2(torch.norm(features, dim=-1), torch.sum(features, dim=-1))
        global_coupling = torch.clamp(self.global_coupling, 0.1, 1.0)
        dt = 0.1

        for _ in range(self.num_steps):
            cos_phases = torch.cos(phases).unsqueeze(-1)
            sin_phases = torch.sin(phases).unsqueeze(-1)
            neighbor_cos = torch.bmm(adjacency, cos_phases).squeeze(-1)
            neighbor_sin = torch.bmm(adjacency, sin_phases).squeeze(-1)
            interaction = torch.atan2(neighbor_sin, neighbor_cos + 1e-8) - phases
            phases = (phases + dt * (frequencies + local_coupling * global_coupling * torch.sin(interaction))) % (2 * math.pi)

        sync_features = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1)
        return phases, sync_features

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        phases, sync_features = self.kuramoto_dynamics(features, adjacency)
        enhanced_features = torch.cat([features, sync_features], dim=-1)
        complex_phases = torch.complex(torch.cos(phases), torch.sin(phases))
        return {
            'sync_features': enhanced_features,
            'flow_predictions': self.phase_to_flow(enhanced_features).squeeze(-1),
            'phases': phases,
            'order_parameter': torch.abs(torch.mean(complex_phases, dim=1)),
            'sync_strength': sync_features
        }


class SpectralModule(nn.Module):
    def __init__(self, feature_dim: int, num_eigenvectors: int = 6):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_eigenvectors = num_eigenvectors
        self.spectral_encoder = nn.Sequential(
            nn.Linear(num_eigenvectors, feature_dim // 2), nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
        self.flow_predictor = nn.Sequential(
            nn.Linear(num_eigenvectors + 1, feature_dim // 4), nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )

    def compute_normalized_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = adjacency.shape
        adjacency = torch.clamp((adjacency + adjacency.transpose(-1, -2)) / 2, 0.0, 1.0)
        degree_inv_sqrt = torch.pow(torch.sum(adjacency, dim=-1) + 1e-6, -0.5)
        D = torch.diag_embed(degree_inv_sqrt)
        return torch.eye(num_nodes, device=adjacency.device).unsqueeze(0) - torch.bmm(torch.bmm(D, adjacency), D)

    def spectral_decomposition(self, laplacian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = laplacian.shape
        eigenvalues_list, eigenvectors_list = [], []

        for b in range(batch_size):
            eigenvals, eigenvecs = torch.linalg.eigh(laplacian[b])

            if torch.isnan(eigenvals).any() or torch.isnan(eigenvecs).any():
                eigenvals = torch.linspace(0, 1, self.num_eigenvectors, device=laplacian.device)
                eigenvecs = torch.eye(num_nodes, device=laplacian.device)[:, :self.num_eigenvectors]
                eigenvalues_list.append(eigenvals)
                eigenvectors_list.append(eigenvecs)
                continue

            sorted_indices = torch.argsort(eigenvals)
            eigenvals = eigenvals[sorted_indices]
            eigenvecs = eigenvecs[:, sorted_indices]
            num_vecs = min(self.num_eigenvectors, num_nodes)
            eigenvals_s = eigenvals[:num_vecs]
            eigenvecs_s = eigenvecs[:, :num_vecs]

            if eigenvals_s.shape[0] < self.num_eigenvectors:
                pad_v = torch.zeros(self.num_eigenvectors - eigenvals_s.shape[0], device=eigenvals.device)
                pad_e = torch.zeros(num_nodes, self.num_eigenvectors - eigenvecs_s.shape[1], device=eigenvecs.device)
                eigenvals_s = torch.cat([eigenvals_s, pad_v])
                eigenvecs_s = torch.cat([eigenvecs_s, pad_e], dim=1)

            eigenvalues_list.append(eigenvals_s)
            eigenvectors_list.append(eigenvecs_s)

        return torch.stack(eigenvalues_list), torch.stack(eigenvectors_list)

    def forward(self, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized_laplacian = self.compute_normalized_laplacian(adjacency)
        eigenvalues, eigenvectors = self.spectral_decomposition(normalized_laplacian)
        spectral_features = self.spectral_encoder(eigenvectors)
        batch_size, num_nodes = eigenvectors.shape[:2]
        spectral_gap = (eigenvalues[:, 1] - eigenvalues[:, 0]).unsqueeze(1).expand(-1, num_nodes)
        flow_input = torch.cat([eigenvectors, spectral_gap.unsqueeze(-1)], dim=-1)
        return {
            'spectral_features': spectral_features,
            'flow_predictions': self.flow_predictor(flow_input).squeeze(-1),
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'spectral_gap': spectral_gap
        }


class AdaptiveFusion(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        total_dim = sum(input_dims)
        self.attention_weights = nn.Sequential(nn.Linear(total_dim, len(input_dims)), nn.Softmax(dim=-1))
        self.feature_transform = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.residual_connection = nn.Linear(input_dims[0], output_dim) if input_dims[0] != output_dim else nn.Identity()

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat(feature_list, dim=-1)
        weights = self.attention_weights(concatenated)
        weighted_features = []
        start_idx = 0
        for i, dim in enumerate(self.input_dims):
            feature_slice = concatenated[..., start_idx:start_idx + dim]
            weighted_features.append(feature_slice * weights[..., i:i+1])
            start_idx += dim
        fused = self.feature_transform(torch.cat(weighted_features, dim=-1))
        return fused + self.residual_connection(feature_list[0])


class MCPST_FSL(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        model_config = config['model']
        self.input_dim = int(model_config['input_dim'])
        self.hidden_dim = int(model_config['hidden_dim'])
        self.pred_len = int(model_config['pred_len'])
        self.spatial_dim = int(model_config['spatial_dim'])
        self.num_nodes = int(config['data'][config['training']['test_dataset']]['node_num'])
        self.node_minibatch_cpu = int(model_config.get('node_minibatch_cpu', 4096))
        self.node_minibatch_gpu = int(model_config.get('node_minibatch_gpu', 1024))

        num_diffusion_steps = int(model_config['num_diffusion_steps'])
        num_oscillator_steps = int(model_config['num_oscillator_steps'])
        num_eigen_vectors = int(model_config['num_eigen_vectors'])

        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.stabilizing_gcn = StabilizingGCN(self.hidden_dim, self.hidden_dim)
        self.thermodynamic_module = ThermodynamicModule(self.hidden_dim, num_diffusion_steps)
        self.kuramoto_module = KuramotoModule(self.hidden_dim, num_oscillator_steps)
        self.spectral_module = SpectralModule(self.hidden_dim, num_eigen_vectors)

        fusion_dims = [self.hidden_dim, self.hidden_dim + 2, self.hidden_dim // 4]
        self.adaptive_fusion = AdaptiveFusion(fusion_dims, self.hidden_dim)

        self.enhanced_temporal_encoder = MultiScaleTemporalEncoder(
            input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, dropout=0.1
        )
        self.horizon_predictor = HorizonSpecificPredictor(
            hidden_dim=self.hidden_dim, num_nodes=self.num_nodes,
            prediction_horizon=self.pred_len, output_dim=1
        )
        self.spatial_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.pred_len * self.spatial_dim)
        )
        self.temporal_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.pred_len)
        )
        self.confidence_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4), nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1), nn.Sigmoid()
        )
        self.physics_weights = nn.Parameter(torch.ones(3) / 3)
        self.final_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def _node_minibatch_size(self, batch_size: int) -> int:
        device = next(self.parameters()).device
        raw = self.node_minibatch_gpu if device.type == 'cuda' else self.node_minibatch_cpu
        return max(1, raw // max(batch_size, 1))

    def _encode_nodes_minibatch(self, fused_reshaped: torch.Tensor, batch_size: int, seq_len: int, num_nodes: int) -> torch.Tensor:
        chunk_size = self._node_minibatch_size(batch_size)
        node_seq = fused_reshaped.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, self.hidden_dim)
        outputs = []
        for start in range(0, batch_size * num_nodes, chunk_size):
            end = min(start + chunk_size, batch_size * num_nodes)
            outputs.append(self.enhanced_temporal_encoder(node_seq[start:end]))
        return torch.cat(outputs, dim=0).view(batch_size, num_nodes, self.hidden_dim)

    def process_adjacency(self, adjacency: torch.Tensor, batch_size: int, seq_len: int, num_nodes: int) -> torch.Tensor:
        if adjacency.dim() == 2:
            if adjacency.shape[0] != num_nodes:
                min_size = min(num_nodes, adjacency.shape[0])
                adjacency = adjacency[:min_size, :min_size]
            adjacency = adjacency.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        elif adjacency.dim() == 3:
            if adjacency.shape[0] == batch_size:
                adjacency = adjacency.unsqueeze(1).expand(-1, seq_len, -1, -1).contiguous().view(batch_size * seq_len, num_nodes, num_nodes)
            else:
                adjacency = adjacency[0].unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        return adjacency

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        if node_features.dim() == 4 and node_features.shape[1] > node_features.shape[2]:
            node_features = node_features.transpose(1, 2)

        batch_size, seq_len, num_nodes, feature_dim = node_features.shape
        min_nodes = min(num_nodes, adjacency_matrix.shape[-1])
        node_features = node_features[:, :, :min_nodes, :]
        num_nodes = min_nodes

        node_features_flat = node_features.reshape(batch_size * seq_len, num_nodes, feature_dim)
        projected_features = self.input_projection(node_features_flat)

        adjacency_expanded = self.process_adjacency(adjacency_matrix, batch_size, seq_len, num_nodes)
        stabilized_features = self.stabilizing_gcn(projected_features, adjacency_expanded)

        heat_results = self.thermodynamic_module(stabilized_features, adjacency_expanded)
        kuramoto_results = self.kuramoto_module(stabilized_features, adjacency_expanded)
        spectral_results = self.spectral_module(adjacency_expanded)

        fused_features = self.adaptive_fusion([
            heat_results['diffused_features'],
            kuramoto_results['sync_features'],
            spectral_results['spectral_features']
        ])

        fused_reshaped = fused_features.view(batch_size, seq_len, num_nodes, -1)
        enhanced_temporal_output = self._encode_nodes_minibatch(fused_reshaped, batch_size, seq_len, num_nodes)

        combined_features = torch.cat([enhanced_temporal_output, torch.mean(fused_reshaped, dim=1)], dim=-1)
        final_features = self.final_fusion(combined_features)

        # HorizonSpecificPredictor now returns (batch, num_nodes, pred_len) directly.
        # physics_flow below is also (batch, num_nodes, pred_len).
        # Transposing here would produce (batch, pred_len, num_nodes) and break the addition.
        flow_predictions, flow_uncertainty = self.horizon_predictor(final_features)
        # flow_predictions: (batch, num_nodes, pred_len)
        # flow_uncertainty: (batch, num_nodes, pred_len)

        spatial_predictions = self.spatial_predictor(final_features).view(batch_size, num_nodes, self.pred_len, self.spatial_dim)
        temporal_predictions = self.temporal_predictor(final_features)
        confidence = self.confidence_predictor(final_features).squeeze(-1)

        physics_weights = F.softmax(self.physics_weights, dim=0)
        physics_flow = (
            physics_weights[0] * heat_results['flow_predictions'].view(batch_size, seq_len, num_nodes)[:, -1, :].unsqueeze(-1).expand(-1, -1, self.pred_len) +
            physics_weights[1] * kuramoto_results['flow_predictions'].view(batch_size, seq_len, num_nodes)[:, -1, :].unsqueeze(-1).expand(-1, -1, self.pred_len) +
            physics_weights[2] * spectral_results['flow_predictions'].view(batch_size, seq_len, num_nodes)[:, -1, :].unsqueeze(-1).expand(-1, -1, self.pred_len)
        )

        final_flow = 0.7 * flow_predictions + 0.3 * physics_flow

        return {
            'flow_predictions': final_flow,        # (batch, num_nodes, pred_len)
            'spatial_predictions': spatial_predictions,
            'temporal_predictions': temporal_predictions,
            'confidence': confidence,
            'physics_weights': physics_weights,
            'uncertainty': flow_uncertainty,       # (batch, num_nodes, pred_len)
            'heat_features': heat_results['diffused_features'].view(batch_size, seq_len, num_nodes, -1),
            'kuramoto_features': kuramoto_results['sync_features'].view(batch_size, seq_len, num_nodes, -1),
            'spectral_features': spectral_results['spectral_features'].view(batch_size, seq_len, num_nodes, -1),
            'fused_features': fused_features.view(batch_size, seq_len, num_nodes, -1),
            'order_parameter': kuramoto_results['order_parameter'].view(batch_size, seq_len),
            'spectral_gap': spectral_results['spectral_gap'][:min(batch_size * seq_len, spectral_results['spectral_gap'].shape[0])].view(batch_size, seq_len, num_nodes)
        }


class PhysicsInformedLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        physics_cfg = config.get('model', {}).get('physics', {})
        self.alpha_flow        = float(physics_cfg.get('alpha_flow',        2.0))
        self.alpha_spatial     = float(physics_cfg.get('alpha_spatial',     0.08))
        self.alpha_temporal    = float(physics_cfg.get('alpha_temporal',    0.08))
        self.alpha_physics     = float(physics_cfg.get('alpha_physics',     0.03))
        self.alpha_consistency = float(physics_cfg.get('alpha_consistency', 0.015))
        self.alpha_uncertainty = float(physics_cfg.get('alpha_uncertainty', 0.01))

        self.flow_loss        = nn.MSELoss()
        self.spatial_loss_fn  = nn.MSELoss()
        self.temporal_loss_fn = nn.MSELoss()
        self.uncertainty_loss = nn.MSELoss()

    @staticmethod
    def _align_flow_shapes(flow_pred: torch.Tensor,
                           flow_target: torch.Tensor) -> torch.Tensor:
        """
        Ensure flow_target has the same layout as flow_pred.

        flow_pred   is always (batch, num_nodes, pred_len) from the model.
        flow_target from the dataset may be:
          • (batch, num_nodes, pred_len)  — new format (correct, no-op)
          • (batch, pred_len, num_nodes)  — old cached format (needs transpose)

        Detection: if shape[1] matches pred's shape[2] AND shape[2] matches
        pred's shape[1] *and* they are not the same value (square would be
        ambiguous, but pred_len=12 ≠ num_nodes=207 in practice).
        """
        if flow_target.dim() != 3 or flow_pred.dim() != 3:
            return flow_target
        if flow_target.shape == flow_pred.shape:
            return flow_target
        if (flow_target.shape[1] == flow_pred.shape[2] and
                flow_target.shape[2] == flow_pred.shape[1]):
            return flow_target.transpose(1, 2).contiguous()
        return flow_target

    @staticmethod
    def _align_uncertainty(uncertainty: torch.Tensor,
                           flow_pred: torch.Tensor) -> torch.Tensor:
        """
        Bring uncertainty to the same (batch, num_nodes, pred_len) layout
        as flow_pred.

        The old HorizonSpecificPredictor stacked uncertainties along dim=1
        giving (batch, pred_len, num_nodes).  The fixed version gives
        (batch, num_nodes, pred_len) directly.  Handle both.
        """
        if uncertainty.dim() == 2:
            return uncertainty.unsqueeze(-1).expand_as(flow_pred)
        if uncertainty.dim() == 3:
            if uncertainty.shape == flow_pred.shape:
                return uncertainty
            # Transpose if (batch, pred_len, num_nodes)
            if (uncertainty.shape[1] == flow_pred.shape[2] and
                    uncertainty.shape[2] == flow_pred.shape[1]):
                uncertainty = uncertainty.transpose(1, 2).contiguous()
        return uncertainty

    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        total_loss = torch.tensor(0.0, device=next(
            (v for v in outputs.values() if isinstance(v, torch.Tensor)), 
            torch.zeros(1)
        ).device, requires_grad=True)
        # Re-initialise as a proper float so += works with tensors
        total_loss = 0.0
        loss_dict = {}

        # ── Flow loss ──────────────────────────────────────────────────────────
        if 'flow' in targets:
            flow_pred = outputs['flow_predictions']   # (batch, num_nodes, pred_len)
            flow_target = targets['flow']

            if flow_target.dim() == 4:
                flow_target = flow_target[..., 0]

            # Align layout: old cached data may be (batch, pred_len, num_nodes)
            flow_target = self._align_flow_shapes(flow_pred, flow_target)

            # Trim to common size (handles node-count differences across datasets)
            min_nodes = min(flow_pred.shape[1], flow_target.shape[1])
            min_time  = min(flow_pred.shape[2], flow_target.shape[2])
            fp = flow_pred[:, :min_nodes, :min_time]
            ft = flow_target[:, :min_nodes, :min_time]

            flow_loss_val = self.flow_loss(fp, ft)
            loss_dict['flow_loss'] = flow_loss_val
            total_loss = total_loss + self.alpha_flow * flow_loss_val

            # ── Uncertainty loss ───────────────────────────────────────────────
            if 'uncertainty' in outputs:
                unc = outputs['uncertainty']           # (batch, num_nodes, pred_len)
                unc = self._align_uncertainty(unc, flow_pred)

                # Trim to same shape as fp/ft
                unc = unc[:, :min_nodes, :min_time]
                target_err = torch.abs(fp - ft).detach()
                unc_loss_val = self.uncertainty_loss(unc, target_err)
                loss_dict['uncertainty_loss'] = unc_loss_val
                total_loss = total_loss + self.alpha_uncertainty * unc_loss_val

        # ── Spatial loss ───────────────────────────────────────────────────────
        if 'spatial' in targets and 'spatial_predictions' in outputs:
            sp = outputs['spatial_predictions']   # (batch, num_nodes, pred_len, spatial_dim)
            st = targets['spatial']               # (batch, num_nodes, pred_len, spatial_dim)
            b = min(sp.shape[0], st.shape[0])
            n = min(sp.shape[1], st.shape[1])
            t = min(sp.shape[2], st.shape[2])
            d = min(sp.shape[3], st.shape[3])
            spatial_loss_val = self.spatial_loss_fn(sp[:b, :n, :t, :d], st[:b, :n, :t, :d])
            loss_dict['spatial_loss'] = spatial_loss_val
            total_loss = total_loss + self.alpha_spatial * spatial_loss_val

        # ── Temporal loss ──────────────────────────────────────────────────────
        if 'temporal' in targets and 'temporal_predictions' in outputs:
            tp = outputs['temporal_predictions']  # (batch, num_nodes, pred_len)
            tt = targets['temporal']              # (batch, num_nodes, pred_len)
            b = min(tp.shape[0], tt.shape[0])
            n = min(tp.shape[1], tt.shape[1])
            t = min(tp.shape[2], tt.shape[2])
            temporal_loss_val = self.temporal_loss_fn(tp[:b, :n, :t], tt[:b, :n, :t])
            loss_dict['temporal_loss'] = temporal_loss_val
            total_loss = total_loss + self.alpha_temporal * temporal_loss_val

        # ── Physics regularisation ─────────────────────────────────────────────
        dev = total_loss.device if isinstance(total_loss, torch.Tensor) else torch.device('cpu')
        physics_loss = torch.zeros(1, device=dev).squeeze()
        if 'order_parameter' in outputs:
            physics_loss = physics_loss + torch.mean((outputs['order_parameter'] - 0.5) ** 2)
        if 'spectral_gap' in outputs:
            physics_loss = physics_loss + torch.mean(
                torch.clamp(outputs['spectral_gap'], 0, 1) ** 2
            ) * 0.1
        loss_dict['physics_loss'] = physics_loss
        total_loss = total_loss + self.alpha_physics * physics_loss

        # ── Consistency regularisation ─────────────────────────────────────────
        consistency_loss = torch.zeros(1, device=dev).squeeze()
        if 'physics_weights' in outputs:
            consistency_loss = torch.mean(
                (torch.sum(outputs['physics_weights']) - 1.0) ** 2
            )
        loss_dict['consistency_loss'] = consistency_loss
        total_loss = total_loss + self.alpha_consistency * consistency_loss

        loss_dict['total_loss'] = total_loss

        # Ensure all expected keys exist (some may be absent if targets lack them)
        for key in ('spatial_loss', 'temporal_loss', 'uncertainty_loss'):
            if key not in loss_dict:
                device = (total_loss.device if isinstance(total_loss, torch.Tensor)
                          else torch.device('cpu'))
                loss_dict[key] = torch.tensor(0.0, device=device)

        return loss_dict