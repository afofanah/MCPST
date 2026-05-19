import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_scales=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        self.shared_lstm = nn.LSTM(input_dim, hidden_dim // 8, batch_first=True, dropout=dropout, num_layers=2)
        
        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_dim // 8, hidden_dim // 8) for _ in range(num_scales)
        ])
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
        
        lstm_concat_dim = num_scales * (hidden_dim // 8)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, lstm_concat_dim) * 0.1)
        self.lstm_projection = nn.Linear(lstm_concat_dim, hidden_dim)
        
        self.seasonal_detector = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=5, padding=2, groups=input_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 8, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        lstm_out, _ = self.shared_lstm(x)
        
        scale_outputs = []
        for i, projection in enumerate(self.scale_projections):
            if i == 0:
                scale_output = projection(lstm_out)
            elif i == 1 and seq_len > 2:
                downsampled = lstm_out[:, ::2, :]
                if downsampled.size(1) != seq_len:
                    downsampled = F.interpolate(
                        downsampled.transpose(1, 2), 
                        size=seq_len, 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                scale_output = projection(downsampled)
            elif i == 2 and seq_len > 4:
                downsampled = lstm_out[:, ::4, :]
                if downsampled.size(1) != seq_len:
                    downsampled = F.interpolate(
                        downsampled.transpose(1, 2), 
                        size=seq_len, 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                scale_output = projection(downsampled)
            else:
                scale_output = projection(lstm_out)
            
            scale_outputs.append(scale_output)
        
        combined_lstm = torch.cat(scale_outputs, dim=-1)
        
        pos_enc = self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        combined_lstm = combined_lstm + pos_enc
        
        combined_lstm = self.lstm_projection(combined_lstm)
        
        transformer_out = self.transformer_encoder(combined_lstm)
        
        memory_attended, _ = self.memory_attention(
            transformer_out, transformer_out, transformer_out
        )
        
        seasonal_features = self.seasonal_detector(x.transpose(1, 2))
        trend_features = self.trend_extractor(x.transpose(1, 2))
        
        memory_pooled = torch.mean(memory_attended, dim=1)
        transformer_pooled = torch.mean(transformer_out, dim=1)
        
        all_features = torch.cat([
            memory_pooled, 
            transformer_pooled, 
            seasonal_features,
            trend_features
        ], dim=-1)
        
        output = self.fusion_layers(all_features)
        output = output + transformer_pooled
        output = self.norm(output)
        
        return output


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
        
        self.final_aggregators = nn.ModuleDict()
        
    def _create_horizon_head(self, hidden_dim, horizon_type):
        if horizon_type == "short":
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU()
            )
        elif horizon_type == "medium":
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU()
            )
        else:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU()
            )
    
    def _create_uncertainty_head(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
            nn.Softplus()
        )
    
    def _get_or_create_aggregator(self, input_size, num_nodes_actual):
        key = f"{input_size}_{num_nodes_actual}"
        
        if key not in self.final_aggregators:
            output_size = num_nodes_actual * self.prediction_horizon * self.output_dim
            self.final_aggregators[key] = nn.Sequential(
                nn.Linear(input_size, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, output_size)
            ).to(next(self.parameters()).device)
        
        return self.final_aggregators[key]
    
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
            
            weighted_features = features * horizon_weights[h]
            combined_features.append(weighted_features)
            uncertainties.append(uncertainty)
        
        aggregated_features = torch.stack(combined_features, dim=1).mean(dim=1)
        aggregated_features = aggregated_features.view(batch_size, -1)
        
        aggregator = self._get_or_create_aggregator(aggregated_features.size(1), num_nodes_actual)
        prediction = aggregator(aggregated_features)
        
        prediction = prediction.view(batch_size, num_nodes_actual, self.prediction_horizon)
        prediction = prediction.transpose(1, 2)
        
        uncertainty = torch.stack(uncertainties, dim=1).squeeze(-1)
        
        return prediction, uncertainty


class StabilizingGCN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.linear_transform = nn.Linear(feature_dim, hidden_dim)
        self.graph_conv = nn.Linear(hidden_dim, hidden_dim)
        self.output_transform = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = features.shape
        
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        
        x = self.linear_transform(features)
        x = torch.bmm(adjacency, x)
        x = self.layer_norm(x)
        x = F.gelu(self.graph_conv(x))
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
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.flow_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
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
            diffusion = dt * conductivity / capacity * gradient
            current_state = current_state + diffusion
            current_state = torch.clamp(current_state, -2.0, 2.0)
        
        return current_state
    
    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        degree = torch.sum(adjacency, dim=-1, keepdim=True) + 1e-6
        degree_matrix = torch.diag_embed(degree.squeeze(-1))
        laplacian = degree_matrix - adjacency
        
        diffused_features = self.heat_diffusion(features, laplacian)
        flow_predictions = self.flow_head(diffused_features).squeeze(-1)
        
        return {
            'diffused_features': diffused_features,
            'flow_predictions': flow_predictions,
            'heat_sources': self.heat_source_predictor(features).squeeze(-1)
        }


class KuramotoModule(nn.Module):
    def __init__(self, feature_dim: int, num_steps: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        
        self.frequency_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.coupling_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.phase_to_flow = nn.Sequential(
            nn.Linear(feature_dim + 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.global_coupling = nn.Parameter(torch.tensor(0.3))
    
    def kuramoto_dynamics(self, features: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, feature_dim = features.shape
        
        frequencies = self.frequency_predictor(features).squeeze(-1)
        local_coupling = self.coupling_predictor(features).squeeze(-1)
        
        phases = torch.atan2(
            torch.norm(features, dim=-1), 
            torch.sum(features, dim=-1)
        )
        
        global_coupling = torch.clamp(self.global_coupling, 0.1, 1.0)
        dt = 0.1
        
        for _ in range(self.num_steps):
            cos_phases = torch.cos(phases).unsqueeze(-1)
            sin_phases = torch.sin(phases).unsqueeze(-1)
            
            neighbor_cos = torch.bmm(adjacency, cos_phases).squeeze(-1)
            neighbor_sin = torch.bmm(adjacency, sin_phases).squeeze(-1)
            
            interaction = torch.atan2(neighbor_sin, neighbor_cos + 1e-8) - phases
            coupling_strength = local_coupling * global_coupling
            
            phase_dot = frequencies + coupling_strength * torch.sin(interaction)
            phases = phases + dt * phase_dot
            phases = phases % (2 * torch.pi)
        
        sync_features = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1)
        
        return phases, sync_features
    
    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        phases, sync_features = self.kuramoto_dynamics(features, adjacency)
        
        enhanced_features = torch.cat([features, sync_features], dim=-1)
        flow_predictions = self.phase_to_flow(enhanced_features).squeeze(-1)
        
        complex_phases = torch.complex(torch.cos(phases), torch.sin(phases))
        order_parameter = torch.abs(torch.mean(complex_phases, dim=1))
        
        return {
            'sync_features': enhanced_features,
            'flow_predictions': flow_predictions,
            'phases': phases,
            'order_parameter': order_parameter,
            'sync_strength': sync_features
        }


class SpectralModule(nn.Module):
    def __init__(self, feature_dim: int, num_eigenvectors: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_eigenvectors = num_eigenvectors
        
        self.spectral_encoder = nn.Sequential(
            nn.Linear(num_eigenvectors, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
        
        self.flow_predictor = nn.Sequential(
            nn.Linear(num_eigenvectors + 1, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def compute_normalized_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = adjacency.shape
        
        adjacency = torch.clamp(adjacency, 0.0, 1.0)
        adjacency = (adjacency + adjacency.transpose(-1, -2)) * 0.5
        
        degree = torch.sum(adjacency, dim=-1) + 1e-6
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_matrix = torch.diag_embed(degree_inv_sqrt)
        
        identity = torch.eye(num_nodes, device=adjacency.device).unsqueeze(0).expand(batch_size, -1, -1)
        normalized_laplacian = identity - torch.bmm(torch.bmm(degree_matrix, adjacency), degree_matrix)
        
        return normalized_laplacian
    
    def spectral_decomposition(self, laplacian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = laplacian.shape
        
        eigenvalues_list = []
        eigenvectors_list = []
        
        for b in range(batch_size):
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(laplacian[b])
                
                if torch.isnan(eigenvals).any() or torch.isnan(eigenvecs).any():
                    eigenvals_subset = torch.linspace(0, 1, self.num_eigenvectors, device=laplacian.device)
                    eigenvecs_subset = torch.eye(num_nodes, device=laplacian.device)[:, :self.num_eigenvectors]
                    eigenvalues_list.append(eigenvals_subset)
                    eigenvectors_list.append(eigenvecs_subset)
                    continue
                
                sorted_indices = torch.argsort(eigenvals)[:self.num_eigenvectors]
                eigenvals_subset = eigenvals[sorted_indices]
                eigenvecs_subset = eigenvecs[:, sorted_indices]
                
                eigenvalues_list.append(eigenvals_subset)
                eigenvectors_list.append(eigenvecs_subset)
                
            except:
                eigenvals_subset = torch.linspace(0, 1, self.num_eigenvectors, device=laplacian.device)
                eigenvecs_subset = torch.eye(num_nodes, device=laplacian.device)[:, :self.num_eigenvectors]
                eigenvalues_list.append(eigenvals_subset)
                eigenvectors_list.append(eigenvecs_subset)
        
        return torch.stack(eigenvalues_list), torch.stack(eigenvectors_list)
    
    def forward(self, adjacency: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized_laplacian = self.compute_normalized_laplacian(adjacency)
        eigenvalues, eigenvectors = self.spectral_decomposition(normalized_laplacian)
        
        spectral_features = self.spectral_encoder(eigenvectors)
        
        batch_size, num_nodes = eigenvectors.shape[:2]
        spectral_gap = (eigenvalues[:, 1] - eigenvalues[:, 0]).unsqueeze(1).expand(-1, num_nodes)
        
        flow_input = torch.cat([eigenvectors, spectral_gap.unsqueeze(-1)], dim=-1)
        flow_predictions = self.flow_predictor(flow_input).squeeze(-1)
        
        return {
            'spectral_features': spectral_features,
            'flow_predictions': flow_predictions,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'spectral_gap': spectral_gap
        }


class AdaptiveFusion(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        total_dim = sum(input_dims)
        
        self.attention_weights = nn.Sequential(
            nn.Linear(total_dim, len(input_dims)),
            nn.Softmax(dim=-1)
        )
        
        self.feature_transform = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        self.residual_connection = nn.Linear(input_dims[0], output_dim) if input_dims[0] != output_dim else nn.Identity()
    
    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat(feature_list, dim=-1)
        
        weights = self.attention_weights(concatenated)
        
        weighted_features = []
        start_idx = 0
        for i, dim in enumerate(self.input_dims):
            end_idx = start_idx + dim
            feature_slice = concatenated[..., start_idx:end_idx]
            weighted_feature = feature_slice * weights[..., i:i+1]
            weighted_features.append(weighted_feature)
            start_idx = end_idx
        
        weighted_concat = torch.cat(weighted_features, dim=-1)
        fused = self.feature_transform(weighted_concat)
        
        residual = self.residual_connection(feature_list[0])
        
        return fused + residual


class PCLAD_Traffic(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        
        model_config = config['model']
        self.input_dim = int(model_config['input_dim'])
        self.hidden_dim = int(model_config['hidden_dim'])
        self.pred_len = int(model_config['pred_len'])
        self.spatial_dim = int(model_config['spatial_dim'])
        self.num_nodes = int(config['data'][config['training']['test_dataset']]['node_num'])
        
        num_diffusion_steps = int(model_config['num_diffusion_steps'])
        num_oscillator_steps = int(model_config['num_oscillator_steps'])
        num_eigen_vectors = int(model_config['num_eigen_vectors'])
        
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.stabilizing_gcn = StabilizingGCN(self.hidden_dim, self.hidden_dim)
        
        self.thermodynamic_module = ThermodynamicModule(
            self.hidden_dim, 
            num_diffusion_steps
        )
        self.kuramoto_module = KuramotoModule(
            self.hidden_dim, 
            num_oscillator_steps
        )
        self.spectral_module = SpectralModule(
            self.hidden_dim, 
            num_eigen_vectors
        )
        
        fusion_dims = [self.hidden_dim, self.hidden_dim + 2, self.hidden_dim // 4]
        self.adaptive_fusion = AdaptiveFusion(fusion_dims, self.hidden_dim)
        
        self.enhanced_temporal_encoder = MultiScaleTemporalEncoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.1
        )
        
        self.horizon_predictor = HorizonSpecificPredictor(
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            prediction_horizon=self.pred_len,
            output_dim=1
        )
        
        self.spatial_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.pred_len * self.spatial_dim)
        )
        
        self.temporal_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.pred_len)
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.physics_weights = nn.Parameter(torch.ones(3) / 3)
        
        self.final_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def process_adjacency(self, adjacency: torch.Tensor, batch_size: int, seq_len: int, num_nodes: int) -> torch.Tensor:
        if adjacency.dim() == 2:
            if adjacency.shape[0] != num_nodes:
                min_size = min(num_nodes, adjacency.shape[0])
                adjacency = adjacency[:min_size, :min_size]
            adjacency = adjacency.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        elif adjacency.dim() == 3:
            if adjacency.shape[0] == batch_size:
                adjacency = adjacency.unsqueeze(1).expand(-1, seq_len, -1, -1)
                adjacency = adjacency.contiguous().view(batch_size * seq_len, num_nodes, num_nodes)
            else:
                adjacency = adjacency[0].unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        
        return adjacency
    
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        if node_features.dim() == 4:
            if node_features.shape[1] > node_features.shape[2]:
                node_features = node_features.transpose(1, 2)
        
        batch_size, seq_len, num_nodes, feature_dim = node_features.shape
        
        min_nodes = min(num_nodes, adjacency_matrix.shape[-1])
        if min_nodes != num_nodes:
            node_features = node_features[:, :, :min_nodes, :]
            num_nodes = min_nodes
        
        node_features_flat = node_features.reshape(batch_size * seq_len, num_nodes, feature_dim)
        projected_features = self.input_projection(node_features_flat)
        
        adjacency_expanded = self.process_adjacency(adjacency_matrix, batch_size, seq_len, num_nodes)
        
        stabilized_features = self.stabilizing_gcn(projected_features, adjacency_expanded)
        
        heat_results = self.thermodynamic_module(stabilized_features, adjacency_expanded)
        kuramoto_results = self.kuramoto_module(stabilized_features, adjacency_expanded)
        spectral_results = self.spectral_module(adjacency_expanded)
        
        feature_list = [
            heat_results['diffused_features'],
            kuramoto_results['sync_features'],
            spectral_results['spectral_features']
        ]
        
        fused_features = self.adaptive_fusion(feature_list)
        
        fused_reshaped = fused_features.view(batch_size, seq_len, num_nodes, -1)
        
        temporal_features = []
        for node_idx in range(num_nodes):
            node_temporal_features = fused_reshaped[:, :, node_idx, :]
            enhanced_node_features = self.enhanced_temporal_encoder(node_temporal_features)
            temporal_features.append(enhanced_node_features)
        
        enhanced_temporal_output = torch.stack(temporal_features, dim=1)
        
        combined_features = torch.cat([enhanced_temporal_output, torch.mean(fused_reshaped, dim=1)], dim=-1)
        final_features = self.final_fusion(combined_features)
        
        flow_predictions, flow_uncertainty = self.horizon_predictor(final_features)
        flow_predictions = flow_predictions.transpose(1, 2)
        
        spatial_predictions = self.spatial_predictor(final_features).view(batch_size, num_nodes, self.pred_len, self.spatial_dim)
        temporal_predictions = self.temporal_predictor(final_features)
        confidence = self.confidence_predictor(final_features).squeeze(-1)
        
        physics_weights = F.softmax(self.physics_weights, dim=0)
        
        physics_flow = (physics_weights[0] * heat_results['flow_predictions'].view(batch_size, seq_len, num_nodes)[:, -1, :].unsqueeze(-1).expand(-1, -1, self.pred_len) +
                       physics_weights[1] * kuramoto_results['flow_predictions'].view(batch_size, seq_len, num_nodes)[:, -1, :].unsqueeze(-1).expand(-1, -1, self.pred_len) +
                       physics_weights[2] * spectral_results['flow_predictions'].view(batch_size, seq_len, num_nodes)[:, -1, :].unsqueeze(-1).expand(-1, -1, self.pred_len))
        
        alpha = 0.7
        final_flow = alpha * flow_predictions + (1 - alpha) * physics_flow
        
        return {
            'flow_predictions': final_flow,
            'spatial_predictions': spatial_predictions,
            'temporal_predictions': temporal_predictions,
            'confidence': confidence,
            'physics_weights': physics_weights,
            'uncertainty': flow_uncertainty,
            'heat_features': heat_results['diffused_features'].view(batch_size, seq_len, num_nodes, -1),
            'kuramoto_features': kuramoto_results['sync_features'].view(batch_size, seq_len, num_nodes, -1),
            'spectral_features': spectral_results['spectral_features'].view(batch_size, seq_len, num_nodes, -1),
            'fused_features': fused_features.view(batch_size, seq_len, num_nodes, -1),
            'order_parameter': kuramoto_results['order_parameter'].view(batch_size, seq_len),
            'spectral_gap': spectral_results['spectral_gap'][:min(batch_size*seq_len, spectral_results['spectral_gap'].shape[0])].view(batch_size, seq_len, num_nodes)
        }


class PhysicsInformedLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.alpha_flow = 2.0
        self.alpha_spatial = 0.08
        self.alpha_temporal = 0.08
        self.alpha_physics = 0.03
        self.alpha_consistency = 0.015
        self.alpha_uncertainty = 0.01
        
        self.flow_loss = nn.MSELoss()
        self.spatial_loss = nn.MSELoss()
        self.temporal_loss = nn.MSELoss()
        self.uncertainty_loss = nn.MSELoss()
    
    def forward(self, outputs: Dict, targets: Dict) -> Dict:
        total_loss = 0.0
        loss_dict = {}
        
        if 'flow' in targets:
            flow_pred = outputs['flow_predictions']
            flow_target = targets['flow']
            
            if flow_target.dim() == 4:
                flow_target = flow_target[..., 0]
            
            min_nodes = min(flow_pred.shape[1], flow_target.shape[1])
            min_time = min(flow_pred.shape[2], flow_target.shape[2])
            
            flow_pred = flow_pred[:, :min_nodes, :min_time]
            flow_target = flow_target[:, :min_nodes, :min_time]
            
            flow_loss = self.flow_loss(flow_pred, flow_target)
            loss_dict['flow_loss'] = flow_loss
            total_loss += self.alpha_flow * flow_loss
            
            if 'uncertainty' in outputs:
                uncertainty = outputs['uncertainty']
                if uncertainty.shape != flow_pred.shape:
                    if uncertainty.dim() == 2:
                        uncertainty = uncertainty.unsqueeze(-1).expand(-1, -1, flow_pred.shape[2])
                    elif uncertainty.dim() == 3 and uncertainty.shape[2] != flow_pred.shape[2]:
                        uncertainty = uncertainty[:, :, :flow_pred.shape[2]]
                
                uncertainty_target = torch.abs(flow_pred - flow_target).detach()
                uncertainty_loss = self.uncertainty_loss(uncertainty, uncertainty_target)
                loss_dict['uncertainty_loss'] = uncertainty_loss
                total_loss += self.alpha_uncertainty * uncertainty_loss
        
        if 'spatial' in targets and 'spatial_predictions' in outputs:
            spatial_pred = outputs['spatial_predictions']
            spatial_target = targets['spatial']
            
            min_batch = min(spatial_pred.shape[0], spatial_target.shape[0])
            min_nodes = min(spatial_pred.shape[1], spatial_target.shape[1])
            min_time = min(spatial_pred.shape[2], spatial_target.shape[2])
            min_dim = min(spatial_pred.shape[3], spatial_target.shape[3])
            
            spatial_pred = spatial_pred[:min_batch, :min_nodes, :min_time, :min_dim]
            spatial_target = spatial_target[:min_batch, :min_nodes, :min_time, :min_dim]
            
            spatial_loss = self.spatial_loss(spatial_pred, spatial_target)
            loss_dict['spatial_loss'] = spatial_loss
            total_loss += self.alpha_spatial * spatial_loss
        
        if 'temporal' in targets and 'temporal_predictions' in outputs:
            temporal_pred = outputs['temporal_predictions']
            temporal_target = targets['temporal']
            
            min_batch = min(temporal_pred.shape[0], temporal_target.shape[0])
            min_nodes = min(temporal_pred.shape[1], temporal_target.shape[1])
            min_time = min(temporal_pred.shape[2], temporal_target.shape[2])
            
            temporal_pred = temporal_pred[:min_batch, :min_nodes, :min_time]
            temporal_target = temporal_target[:min_batch, :min_nodes, :min_time]
            
            temporal_loss = self.temporal_loss(temporal_pred, temporal_target)
            loss_dict['temporal_loss'] = temporal_loss
            total_loss += self.alpha_temporal * temporal_loss
        
        physics_loss = 0.0
        if 'order_parameter' in outputs:
            order_param = outputs['order_parameter']
            physics_loss += torch.mean((order_param - 0.5) ** 2)
        
        if 'spectral_gap' in outputs:
            spectral_gap = outputs['spectral_gap']
            physics_loss += torch.mean(torch.clamp(spectral_gap, 0, 1) ** 2) * 0.1
        
        loss_dict['physics_loss'] = physics_loss
        total_loss += self.alpha_physics * physics_loss
        
        consistency_loss = 0.0
        if 'physics_weights' in outputs:
            weights = outputs['physics_weights']
            consistency_loss = torch.mean((torch.sum(weights) - 1.0) ** 2)
        
        loss_dict['consistency_loss'] = consistency_loss
        total_loss += self.alpha_consistency * consistency_loss
        
        loss_dict['total_loss'] = total_loss
        
        for key in ['spatial_loss', 'temporal_loss', 'uncertainty_loss']:
            if key not in loss_dict:
                loss_dict[key] = torch.tensor(0.0, device=total_loss.device)
        
        return loss_dict


MCPST_Traffic = MCPST_Traffic
PhysicsInformedLoss = PhysicsInformedLoss
