import asyncio
import json
import time
import logging
import random
from typing import Dict, Optional, Tuple, Any
from collections import deque
from contextlib import nullcontext
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn: Module = object; Sequential = object; Linear = object; ReLU = object; MSELoss = object; Dropout = object; LayerNorm = object; TransformerEncoderLayer = object; TransformerEncoder = object
    class torch: Tensor = object; cat = staticmethod(lambda *a, **kw: None); stack = staticmethod(lambda *a, **kw: None); tensor = staticmethod(lambda *a, **kw: None); float32 = None; no_grad = staticmethod(lambda: nullcontext()); zeros = staticmethod(lambda *a, **kw: None); zeros_like = staticmethod(lambda *a, **kw: None); unsqueeze = staticmethod(lambda *a, **kw: None); squeeze = staticmethod(lambda *a, **kw: None); detach = staticmethod(lambda *a, **kw: None); cpu = staticmethod(lambda *a, **kw: a[0]); numpy = staticmethod(lambda *a, **kw: None)
    class optim: Adam = object
    np = None
    print("WARNING: PyTorch not found. DreamerV3Bubble will be disabled.", file=sys.stderr)

from bubbles_core import UniversalBubble, Actions, Event, UniversalCode, Tags, SystemContext, logger, EventService

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class STORMWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_categories=32, num_classes=32, num_layers=2, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.input_dim = num_categories * num_classes + action_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, num_categories * num_classes)
        )
        self.action_mixer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.positional_encoding = nn.Parameter(torch.randn(1, 10, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, state_dim)
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 41)
        )
        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.dynamics_predictor = nn.Linear(hidden_dim, num_categories * num_classes)

    def forward(self, states, actions, device=torch.device("cpu")):
        batch_size, seq_len, state_dim = states.shape
        logger.debug(f"STORMWorldModel forward: states.shape={states.shape}, actions.shape={actions.shape}")
        state_logits = self.state_encoder(states.to(device)).view(batch_size, seq_len, self.num_categories, self.num_classes)
        logger.debug(f"state_logits.shape={state_logits.shape}")
        state_logits = 0.99 * state_logits + 0.01 * torch.ones_like(state_logits) / self.num_classes
        dist = torch.distributions.Categorical(logits=state_logits)
        z_indices = dist.sample()
        logger.debug(f"z_indices.shape={z_indices.shape}")
        z_one_hot = torch.zeros(batch_size, seq_len, self.num_categories, self.num_classes, device=device)
        z_one_hot.scatter_(3, z_indices.unsqueeze(-1), 1.0)
        z_t = z_one_hot.view(batch_size, seq_len, -1)
        logger.debug(f"z_t.shape={z_t.shape}")
        combined_input = torch.cat([z_t, actions.to(device)], dim=-1)
        logger.debug(f"action_mixer input: combined_input.shape={combined_input.shape}")
        if combined_input.shape[-1] != self.input_dim:
            logger.error(f"Input dimension mismatch: got {combined_input.shape[-1]}, expected {self.input_dim}")
            raise ValueError(f"Input dimension mismatch: got {combined_input.shape[-1]}, expected {self.input_dim}")
        e_t = self.action_mixer(combined_input)
        e_t = e_t + self.positional_encoding[:, :seq_len, :].to(device)
        h = self.transformer(e_t)
        next_state = self.decoder(h[:, -1, :])
        reward_logits = self.reward_predictor(h[:, -1, :])
        continuation = self.continuation_predictor(h[:, -1, :])
        dynamics_logits = self.dynamics_predictor(h[:, -1, :]).view(batch_size, self.num_categories, self.num_classes)
        dynamics_dist = torch.distributions.Categorical(logits=dynamics_logits)
        last_state_logits = state_logits[:, -1, :, :]
        kl_loss = torch.mean(torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=last_state_logits),
            dynamics_dist
        ))
        return next_state, reward_logits, continuation, h[:, -1, :], kl_loss, z_t[:, -1, :]

    def _symlog(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

class DreamerV3Bubble(UniversalBubble):
    def __init__(self, object_id: str, context: SystemContext, state_dim: int = 24, action_dim: int = 5, hidden_dim: int = 512, num_categories: int = 32, horizon: int = 16, num_transformer_layers: int = 2, num_heads: int = 8, dropout_rate: float = 0.1, weight_decay: float = 0.001, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        self.state_dim = 24
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        self.horizon = horizon
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.replay_buffer = deque(maxlen=10000)
        self.validation_buffer = deque(maxlen=1000)
        self.batch_size = 32
        self.sequence_length = 10
        self.learning_rate = 1e-4
        self.entropy_coeff = 3e-4
        self.execution_count = 0
        self.nan_inf_count = 0
        self.training_metrics = {
            "state_loss": 0.0, "reward_loss": 0.0, "kl_loss": 0.0,
            "continuation_loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0,
            "entropy": 0.0, "disagreement_loss": 0.0, "recon_loss": 0.0,
            "avg_return": 0.0, "validation_loss": 0.0
        }
        self.return_range = None
        self.ema_alpha = 0.98
        self.current_known_state: Optional[Dict[str, Any]] = None
        self.state_action_history: Dict[str, Tuple[Dict, Dict]] = {}
        self.state_history = deque(maxlen=20)

        if not TORCH_AVAILABLE:
            logger.error(f"{self.object_id}: PyTorch unavailable, switching to placeholder mode.")
            self.world_model = None
            self.world_model_ensemble = None
            self.actor = None
            self.critic = None
            self.critic_ema = None
            self.device = None
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.world_model = STORMWorldModel(
                self.state_dim, self.action_dim, self.hidden_dim, self.num_categories,
                num_layers=self.num_transformer_layers, num_heads=self.num_heads, dropout_rate=self.dropout_rate
            )
            self.world_model_ensemble = [
                STORMWorldModel(
                    self.state_dim, self.action_dim, self.hidden_dim, self.num_categories,
                    num_layers=self.num_transformer_layers, num_heads=self.num_heads, dropout_rate=self.dropout_rate
                ) for _ in range(3)
            ]
            self.actor = self._build_actor()
            self.critic = self._build_critic()
            self.critic_ema = self._build_critic()
            self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.ensemble_optimizers = [optim.Adam(m.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) for m in self.world_model_ensemble]
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.world_model.to(self.device)
            for m in self.world_model_ensemble:
                m.to(self.device)
            self.actor.to(self.device)
            self.critic.to(self.device)
            self.critic_ema.to(self.device)
            nn.init.zeros_(self.world_model.reward_predictor[-1].weight)
            nn.init.zeros_(self.critic.net[-1].weight)
            self._update_critic_ema(1.0)
            self.replay_buffer.clear()
            self.validation_buffer.clear()

        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: Initialized DreamerV3 (Mode: {'NN' if TORCH_AVAILABLE else 'Placeholder'}, Device: {self.device if TORCH_AVAILABLE else 'None'}, Transformer Layers: {self.num_transformer_layers}, Heads: {self.num_heads})")

    def _symlog(self, x):
        if not TORCH_AVAILABLE:
            return x
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def _symexp(self, x):
        if not TORCH_AVAILABLE:
            return x
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def _twohot_encode(self, y, bins):
        if not TORCH_AVAILABLE:
            return torch.zeros(y.size(0), len(bins))
        if y.dim() > 1:
            y = y.squeeze()
        y = y.clamp(bins[0], bins[-1])
        idx = torch.searchsorted(bins, y)
        idx = idx.clamp(0, len(bins) - 2)
        lower = bins[idx]
        upper = bins[idx + 1]
        weight = (y - lower) / (upper - lower + 1e-8)
        twohot = torch.zeros(y.size(0), len(bins), device=self.device)
        twohot.scatter_(1, idx.unsqueeze(1), 1.0 - weight.unsqueeze(1))
        twohot.scatter_(1, (idx + 1).unsqueeze(1), weight.unsqueeze(1))
        return twohot

    def _build_actor(self):
        if not TORCH_AVAILABLE:
            return None
        class Actor(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim, device, dropout_rate):
                super().__init__()
                self.device = device
                self.hidden_dim = hidden_dim
                self.net = nn.Sequential(
                    nn.Linear(state_dim + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    ResidualBlock(hidden_dim, dropout_rate),
                    nn.Linear(hidden_dim, action_dim)
                )

            def forward(self, state, hidden=None):
                if hidden is not None:
                    state = torch.cat([self._symlog(state), hidden], dim=-1)
                else:
                    padding = torch.zeros(state.shape[0], self.hidden_dim, device=self.device)
                    state = torch.cat([self._symlog(state), padding], dim=-1)
                logger.debug(f"Actor forward: state.shape={state.shape}")
                logits = self.net(state.to(self.device))
                return torch.distributions.Categorical(logits=logits)

            def _symlog(self, x):
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        return Actor(self.state_dim, self.action_dim, self.hidden_dim, self.device, self.dropout_rate)

    def _build_critic(self):
        if not TORCH_AVAILABLE:
            return None
        class Critic(nn.Module):
            def __init__(self, state_dim, hidden_dim, device, dropout_rate):
                super().__init__()
                self.device = device
                self.hidden_dim = hidden_dim
                self.net = nn.Sequential(
                    nn.Linear(state_dim + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    ResidualBlock(hidden_dim, dropout_rate),
                    nn.Linear(hidden_dim, 41)
                )

            def forward(self, state, hidden=None):
                if hidden is not None:
                    state = torch.cat([self._symlog(state), hidden], dim=-1)
                else:
                    padding = torch.zeros(state.shape[0], self.hidden_dim, device=self.device)
                    state = torch.cat([self._symlog(state), padding], dim=-1)
                logits = self.net(state.to(self.device))
                return torch.distributions.Categorical(logits=logits)

            def _symlog(self, x):
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        return Critic(self.state_dim, self.hidden_dim, self.device, self.dropout_rate)

    def _update_critic_ema(self, alpha=0.02):
        if not TORCH_AVAILABLE:
            return
        for param, ema_param in zip(self.critic.parameters(), self.critic_ema.parameters()):
            ema_param.data.mul_(1.0 - alpha).add_(param.data, alpha=alpha)

    async def _subscribe_to_events(self):
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.ACTION_TAKEN, self.handle_event)
            await EventService.subscribe(Actions.PREDICT_STATE_QUERY, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to SYSTEM_STATE_UPDATE, ACTION_TAKEN, PREDICT_STATE_QUERY")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    def _vectorize_state(self, state: Dict) -> Optional[torch.Tensor]:
        if not TORCH_AVAILABLE:
            logger.warning(f"{self.object_id}: Cannot vectorize state, PyTorch unavailable.")
            return None
        try:
            self.state_history.append(state)
            metrics = state.get("metrics", {})
            event_frequencies = state.get("event_frequencies", {})
            perturbation = state.get("response_time_perturbation", 0.0)

            energy_avg = sum(s.get("energy", 0) for s in self.state_history) / len(self.state_history)
            cpu_avg = sum(s.get("cpu_percent", 0) for s in self.state_history) / len(self.state_history)
            memory_avg = sum(s.get("memory_percent", 0) for s in self.state_history) / len(self.state_history)
            energy_var = sum((s.get("energy", 0) - energy_avg) ** 2 for s in self.state_history) / len(self.state_history)
            cpu_var = sum((s.get("cpu_percent", 0) - cpu_avg) ** 2 for s in self.state_history) / len(self.state_history)
            memory_var = sum((s.get("memory_percent", 0) - memory_avg) ** 2 for s in self.state_history) / len(self.state_history)
            trend_length = min(5, len(self.state_history))
            recent_states = list(self.state_history)[-trend_length:]
            energy_trend = (recent_states[-1].get("energy", 0) - recent_states[0].get("energy", 0)) / trend_length if trend_length > 1 else 0
            cpu_trend = (recent_states[-1].get("cpu_percent", 0) - recent_states[0].get("cpu_percent", 0)) / trend_length if trend_length > 1 else 0

            vector = [
                state.get("energy", 0) / 10000.0,
                state.get("cpu_percent", 0) / 100.0,
                state.get("memory_percent", 0) / 100.0,
                state.get("num_bubbles", 0) / 20.0,
                metrics.get("avg_llm_response_time_ms", 0) / 60000.0 * (1 + perturbation),
                metrics.get("code_update_count", 0) / 100.0,
                metrics.get("prediction_cache_hit_rate", 0),
                event_frequencies.get("LLM_QUERY_freq_per_min", 0) / 60.0,
                event_frequencies.get("CODE_UPDATE_freq_per_min", 0) / 10.0,
                event_frequencies.get("ACTION_TAKEN_freq_per_min", 0) / 60.0,
                state.get("gravity_force", 0.0) / 10.0,
                state.get("gravity_direction", 0.0) / 360.0,
                state.get("bubble_pos_x", 0.0) / 100.0,
                state.get("bubble_pos_y", 0.0) / 100.0,
                state.get("cluster_id", 0) / 10.0,
                state.get("cluster_strength", 0.0) / 1.0,
                energy_avg / 10000.0,
                cpu_avg / 100.0,
                memory_avg / 100.0,
                energy_var / 100000.0,
                cpu_var / 100.0,
                memory_var / 100.0,
                energy_trend / 1000.0,
                cpu_trend / 10.0
            ]

            if len(vector) != self.state_dim:
                logger.error(f"{self.object_id}: State vector dimension mismatch: got {len(vector)}, expected {self.state_dim}")
                return None

            tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)
            noise = torch.normal(mean=0.0, std=0.01, size=tensor.shape, device=self.device)
            tensor = tensor + noise
            logger.debug(f"_vectorize_state: tensor.shape={tensor.shape}")
            return self._symlog(tensor)
        except Exception as e:
            logger.error(f"{self.object_id}: Error vectorizing state: {e}", exc_info=True)
            return None

    def _rebuild_models(self):
        if not TORCH_AVAILABLE:
            return
        self.replay_buffer.clear()
        self.validation_buffer.clear()
        self.world_model = STORMWorldModel(
            self.state_dim, self.action_dim, self.hidden_dim, self.num_categories,
            num_layers=self.num_transformer_layers, num_heads=self.num_heads, dropout_rate=self.dropout_rate
        )
        self.world_model_ensemble = [
            STORMWorldModel(
                self.state_dim, self.action_dim, self.hidden_dim, self.num_categories,
                num_layers=self.num_transformer_layers, num_heads=self.num_heads, dropout_rate=self.dropout_rate
            ) for _ in range(3)
        ]
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.critic_ema = self._build_critic()
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.ensemble_optimizers = [optim.Adam(m.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) for m in self.world_model_ensemble]
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.world_model.to(self.device)
        for m in self.world_model_ensemble:
            m.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_ema.to(self.device)
        nn.init.zeros_(self.world_model.reward_predictor[-1].weight)
        nn.init.zeros_(self.critic.net[-1].weight)
        logger.info(f"{self.object_id}: Rebuilt models with state_dim={self.state_dim}, transformer_layers={self.num_transformer_layers}, heads={self.num_heads}")
        logger.info(f"Actor input dim: {self.actor.net[0].in_features}, expected: {self.state_dim + self.hidden_dim}")

    def _vectorize_action(self, action: Dict) -> Optional[torch.Tensor]:
        if not TORCH_AVAILABLE:
            logger.warning(f"{self.object_id}: Cannot vectorize action, PyTorch unavailable.")
            return None
        try:
            action_type_str = action.get("action_type", Actions.ACTION_TYPE_NO_OP.name)
            action_types_ordered = [
                Actions.ACTION_TYPE_CODE_UPDATE.name, Actions.ACTION_TYPE_SELF_QUESTION.name,
                Actions.ACTION_TYPE_SPAWN_BUBBLE.name, Actions.ACTION_TYPE_DESTROY_BUBBLE.name,
                Actions.ACTION_TYPE_NO_OP.name
            ]
            if len(action_types_ordered) != self.action_dim:
                raise ValueError(f"Action dimension mismatch: Code expects {len(action_types_ordered)}, configured {self.action_dim}")
            vector = [1.0 if action_type_str == at else 0.0 for at in action_types_ordered]
            return torch.tensor(vector, dtype=torch.float32).to(self.device)
        except Exception as e:
            logger.error(f"{self.object_id}: Error vectorizing action: {e}", exc_info=True)
            return None

    def _devectorize_state(self, state_vector: torch.Tensor) -> Dict:
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        try:
            vec = self._symexp(state_vector).detach().cpu().numpy()
            if len(vec) != self.state_dim:
                return {"error": f"State vector dim mismatch: got {len(vec)}, expected {self.state_dim}"}
            state = {
                "energy": vec[0] * 10000.0,
                "cpu_percent": max(0.0, min(100.0, vec[1] * 100.0)),
                "memory_percent": max(0.0, min(100.0, vec[2] * 100.0)),
                "num_bubbles": max(0, int(round(vec[3] * 20.0))),
                "metrics": {
                    "avg_llm_response_time_ms": max(0.0, vec[4] * 60000.0),
                    "code_update_count": max(0, int(round(vec[5] * 100.0))),
                    "prediction_cache_hit_rate": max(0.0, min(1.0, vec[6])),
                },
                "event_frequencies": {
                    "LLM_QUERY_freq_per_min": max(0.0, vec[7] * 60.0),
                    "CODE_UPDATE_freq_per_min": max(0.0, vec[8] * 10.0),
                    "ACTION_TAKEN_freq_per_min": max(0.0, vec[9] * 60.0),
                },
                "gravity_force": max(0.0, vec[10] * 10.0),
                "gravity_direction": max(0.0, min(360.0, vec[11] * 360.0)),
                "bubble_pos_x": vec[12] * 100.0,
                "bubble_pos_y": vec[13] * 100.0,
                "cluster_id": max(0, int(round(vec[14] * 10.0))),
                "cluster_strength": max(0.0, min(1.0, vec[15])),
                "energy_avg": vec[16] * 10000.0,
                "cpu_avg": vec[17] * 100.0,
                "memory_avg": vec[18] * 100.0,
                "energy_var": vec[19] * 100000.0,
                "cpu_var": vec[20] * 100.0,
                "memory_var": vec[21] * 100.0,
                "energy_trend": vec[22] * 1000.0,
                "cpu_trend": vec[23] * 10.0,
                "timestamp": time.time(),
                "categorical_confidence": 0.7,
                "continuation_probability": 0.9,
                "response_time_perturbation": 0.1
            }
            return state
        except Exception as e:
            logger.error(f"{self.object_id}: Error devectorizing state: {e}", exc_info=True)
            return {"error": f"State devectorization failed: {e}"}

    def _devectorize_action(self, action_vector: torch.Tensor) -> str:
        if not TORCH_AVAILABLE:
            return Actions.ACTION_TYPE_NO_OP.name
        try:
            action_types_ordered = [
                Actions.ACTION_TYPE_CODE_UPDATE.name, Actions.ACTION_TYPE_SELF_QUESTION.name,
                Actions.ACTION_TYPE_SPAWN_BUBBLE.name, Actions.ACTION_TYPE_DESTROY_BUBBLE.name,
                Actions.ACTION_TYPE_NO_OP.name
            ]
            action_idx = torch.argmax(action_vector).item()
            return action_types_ordered[action_idx]
        except Exception as e:
            logger.error(f"{self.object_id}: Error devectorizing action: {e}", exc_info=True)
            return Actions.ACTION_TYPE_NO_OP.name

    def _simulate_state(self) -> Dict[str, Any]:
        state = {
            "energy": random.uniform(5000, 15000),
            "cpu_percent": random.uniform(5, 80),
            "memory_percent": random.uniform(20, 90),
            "num_bubbles": random.randint(1, 20),
            "metrics": {
                "avg_llm_response_time_ms": random.uniform(100, 5000),
                "code_update_count": random.randint(0, 50),
                "prediction_cache_hit_rate": random.uniform(0, 1),
            },
            "event_frequencies": {
                "LLM_QUERY_freq_per_min": random.uniform(0, 60),
                "CODE_UPDATE_freq_per_min": random.uniform(0, 10),
                "ACTION_TAKEN_freq_per_min": random.uniform(0, 60),
            },
            "gravity_force": random.uniform(0, 10),
            "gravity_direction": random.uniform(0, 360),
            "bubble_pos_x": random.uniform(-100, 100),
            "bubble_pos_y": random.uniform(-100, 100),
            "cluster_id": random.randint(0, 10),
            "cluster_strength": random.uniform(0, 1),
            "response_time_perturbation": random.uniform(-0.1, 0.1),
            "timestamp": time.time()
        }
        if self.state_history:
            prev_state = self.state_history[-1]
            energy_trend = prev_state.get("energy", 10000) * random.uniform(-0.05, 0.05)
            state["energy"] = max(0, min(20000, state["energy"] + energy_trend))
            cpu_load = prev_state.get("cpu_percent", 50) * random.uniform(0.9, 1.1)
            state["cpu_percent"] = max(0, min(100, cpu_load))
            memory_load = prev_state.get("memory_percent", 50) * random.uniform(0.95, 1.05)
            state["memory_percent"] = max(0, min(100, memory_load))
            state["num_bubbles"] = max(1, min(20, prev_state.get("num_bubbles", 5) + random.randint(-1, 1)))
        return state

    def _simulate_next_state(self, current_state: Dict[str, Any], action: Dict) -> Dict[str, Any]:
        next_state = current_state.copy()
        action_type = action.get("action_type", Actions.ACTION_TYPE_NO_OP.name)
        metrics = next_state.get("metrics", {})
        event_frequencies = next_state.get("event_frequencies", {})
        system_load = random.uniform(0.8, 1.2)
        momentum = 0.9 if self.state_history else 1.0

        if action_type == Actions.ACTION_TYPE_CODE_UPDATE.name:
            next_state["energy"] = max(0, next_state["energy"] - 200 * system_load)
            next_state["cpu_percent"] = min(100, next_state["cpu_percent"] + 10 * system_load)
            metrics["code_update_count"] = metrics.get("code_update_count", 0) + 1
            event_frequencies["CODE_UPDATE_freq_per_min"] = event_frequencies.get("CODE_UPDATE_freq_per_min", 0) + 1
        elif action_type == Actions.ACTION_TYPE_SELF_QUESTION.name:
            next_state["cpu_percent"] = min(100, next_state["cpu_percent"] + 5 * system_load)
            metrics["avg_llm_response_time_ms"] = metrics.get("avg_llm_response_time_ms", 1000) * random.uniform(0.9, 1.1)
            event_frequencies["LLM_QUERY_freq_per_min"] = event_frequencies.get("LLM_QUERY_freq_per_min", 0) + 2
        elif action_type == Actions.ACTION_TYPE_SPAWN_BUBBLE.name:
            next_state["cpu_percent"] = min(100, next_state["cpu_percent"] + 15 * system_load)
            next_state["memory_percent"] = min(100, next_state["memory_percent"] + 10 * system_load)
            next_state["num_bubbles"] = min(20, next_state["num_bubbles"] + 1)
            next_state["energy"] = max(0, next_state["energy"] - 100 * system_load)
        elif action_type == Actions.ACTION_TYPE_DESTROY_BUBBLE.name:
            next_state["cpu_percent"] = max(0, next_state["cpu_percent"] - 10 * system_load)
            next_state["memory_percent"] = max(0, next_state["memory_percent"] - 8 * system_load)
            next_state["num_bubbles"] = max(1, next_state["num_bubbles"] - 1)
        elif action_type == Actions.ACTION_TYPE_NO_OP.name:
            pass

        next_state["energy"] = max(0, min(20000, next_state["energy"] + random.uniform(-50, 50) * momentum))
        next_state["cpu_percent"] = max(0, min(100, next_state["cpu_percent"] * momentum + random.uniform(-5, 5) * system_load))
        next_state["memory_percent"] = max(0, min(100, next_state["memory_percent"] * momentum + random.uniform(-3, 3) * system_load))
        next_state["gravity_force"] = max(0, min(10, next_state["gravity_force"] + random.uniform(-1, 1)))
        next_state["gravity_direction"] = (next_state["gravity_direction"] + random.uniform(-10, 10)) % 360
        next_state["bubble_pos_x"] += random.uniform(-5, 5)
        next_state["bubble_pos_y"] += random.uniform(-5, 5)
        next_state["cluster_strength"] = max(0, min(1, next_state["cluster_strength"] + random.uniform(-0.1, 0.1)))
        next_state["response_time_perturbation"] = random.uniform(-0.1, 0.1)
        next_state["timestamp"] = time.time()
        next_state["metrics"] = metrics
        next_state["event_frequencies"] = event_frequencies
        metrics["avg_llm_response_time_ms"] = max(100, metrics.get("avg_llm_response_time_ms", 1000) * random.uniform(0.95, 1.05))
        metrics["prediction_cache_hit_rate"] = max(0, min(1, metrics.get("prediction_cache_hit_rate", 0.5) + random.uniform(-0.05, 0.05)))
        return next_state

    def _compute_reward(self, prev_state: Dict, next_state: Dict) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            return torch.tensor([0.0])
        energy_reward = (next_state.get("energy", 0) - prev_state.get("energy", 0)) / 1000.0
        cpu_penalty = -0.05 * (next_state.get("cpu_percent", 0) / 100.0)
        memory_penalty = -0.05 * (next_state.get("memory_percent", 0) / 100.0)
        cache_hit_rate = next_state.get("metrics", {}).get("prediction_cache_hit_rate", 0)
        cache_bonus = 0.1 * cache_hit_rate
        energy_var = next_state.get("energy_var", 0) / 100000.0
        stability_penalty = -0.01 * energy_var
        total_reward = energy_reward + cpu_penalty + memory_penalty + cache_bonus + stability_penalty
        total_reward = max(-10.0, min(10.0, total_reward))
        logger.debug(f"{self.object_id}: Computed reward: {total_reward:.4f} (energy: {energy_reward:.4f}, cpu: {cpu_penalty:.4f}, memory: {memory_penalty:.4f}, cache: {cache_bonus:.4f}, stability: {stability_penalty:.4f})")
        return torch.tensor([total_reward], dtype=torch.float32).to(self.device)

    def check_training_stability(self, loss: torch.Tensor) -> bool:
        if not TORCH_AVAILABLE:
            return False
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_inf_count += 1
            logger.error(f"{self.object_id}: Detected unstable training (NaN/Inf loss). Consecutive count: {self.nan_inf_count}")
            if self.nan_inf_count > 2:
                self._rebuild_models()
                self.nan_inf_count = 0
                logger.info(f"{self.object_id}: Rebuilt models due to repeated NaN/Inf losses.")
            else:
                self.learning_rate *= 0.5
                self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                for i, m in enumerate(self.world_model_ensemble):
                    self.ensemble_optimizers[i] = optim.Adam(m.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                logger.info(f"{self.object_id}: Reduced learning rate to {self.learning_rate}")
            return False
        self.nan_inf_count = 0
        return True

    def collect_transitions(self, num_transitions: int):
        for _ in range(num_transitions):
            current_state = self._simulate_state()
            action_types = [
                {"action_type": Actions.ACTION_TYPE_CODE_UPDATE.name},
                {"action_type": Actions.ACTION_TYPE_SELF_QUESTION.name},
                {"action_type": Actions.ACTION_TYPE_SPAWN_BUBBLE.name},
                {"action_type": Actions.ACTION_TYPE_DESTROY_BUBBLE.name},
                {"action_type": Actions.ACTION_TYPE_NO_OP.name}
            ]
            action = random.choice(action_types)
            next_state = self._simulate_next_state(current_state, action)
            reward = self._compute_reward(current_state, next_state)
            state_vec = self._vectorize_state(current_state)
            action_vec = self._vectorize_action(action)
            next_state_vec = self._vectorize_state(next_state)
            if state_vec is None or action_vec is None or next_state_vec is None:
                logger.warning(f"{self.object_id}: Skipping invalid simulated transition.")
                continue
            # Check for NaN/Inf in transition tensors
            tensors = {"state_vec": state_vec, "action_vec": action_vec, "reward": reward, "next_state_vec": next_state_vec}
            for name, tensor in tensors.items():
                if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                    logger.warning(f"{self.object_id}: Skipping transition due to NaN/Inf in {name} (shape={tensor.shape})")
                    break
            else:
                logger.debug(f"collect_transitions: state_vec.shape={state_vec.shape}, action_vec.shape={action_vec.shape}")
                transition = (state_vec, action_vec, reward, next_state_vec)
                if random.random() < 0.8:
                    self.replay_buffer.append(transition)
                else:
                    self.validation_buffer.append(transition)
        logger.info(f"{self.object_id}: Collected {num_transitions} transitions. Replay buffer: {len(self.replay_buffer)}, Validation buffer: {len(self.validation_buffer)}")

    async def compute_validation_loss(self):
        if not TORCH_AVAILABLE or not self.world_model or len(self.validation_buffer) == 0:
            return 0.0
        try:
            batch_size = min(self.batch_size, len(self.validation_buffer))
            batch = random.sample(self.validation_buffer, batch_size)
            states, actions, rewards, next_states = zip(*batch)
            states_tensor = torch.stack(states).to(self.device)
            actions_tensor = torch.stack(actions).to(self.device)
            rewards_tensor = torch.stack(rewards).to(self.device)
            next_states_tensor = torch.stack(next_states).to(self.device)

            self.world_model.eval()
            with torch.no_grad():
                predicted_next_states, reward_logits, predicted_continuations, _, kl_loss, _ = self.world_model(
                    states_tensor.unsqueeze(1), actions_tensor.unsqueeze(1), device=self.device
                )
                state_loss = torch.mean((predicted_next_states - next_states_tensor) ** 2)
                bins = self._symexp(torch.linspace(-20, 20, 41).to(self.device))
                rewards_tensor = torch.clamp(rewards_tensor, min=-20.0, max=20.0)
                reward_targets = self._twohot_encode(rewards_tensor, bins)
                reward_loss = torch.nn.functional.cross_entropy(reward_logits, reward_targets, reduction='mean')
                continuation_tensor = torch.ones(batch_size, 1, dtype=torch.float32).to(self.device)
                continuation_loss = nn.BCELoss()(predicted_continuations, continuation_tensor)
                total_loss = state_loss + reward_loss + continuation_loss + kl_loss
                logger.debug(f"{self.object_id}: Validation loss components: state={state_loss.item():.4f}, reward={reward_loss.item():.4f}, continuation={continuation_loss.item():.4f}, kl={kl_loss.item():.4f}")
                return total_loss.item()
        except Exception as e:
            logger.error(f"{self.object_id}: Validation loss computation error: {e}", exc_info=True)
            return 0.0

    async def train_actor_critic(self):
        if not TORCH_AVAILABLE or not self.actor or not self.critic or len(self.replay_buffer) < self.batch_size:
            logger.debug(f"{self.object_id}: Skipping actor-critic training (Torch: {TORCH_AVAILABLE}, Buffer: {len(self.replay_buffer)})")
            return
        try:
            batch_size = min(self.batch_size, len(self.replay_buffer))
            batch = random.sample(self.replay_buffer, batch_size)
            states, actions = zip(*[(s, a) for s, a, _, _ in batch])
            states_tensor = torch.stack(states).to(self.device)
            actions_tensor = torch.stack(actions).to(self.device)
            logger.debug(f"train_actor_critic: states_tensor.shape={states_tensor.shape}, actions_tensor.shape={actions_tensor.shape}")
            self.world_model.eval()
            self.actor.train()
            self.critic.train()

            imagined_states, imagined_rewards, imagined_continuations, actions_taken, hidden_states = [], [], [], [], []
            current_state = states_tensor.unsqueeze(1)
            for step in range(self.horizon):
                action_dist = self.actor(current_state[:, -1, :], hidden_states[-1] if hidden_states else None)
                action = action_dist.sample().unsqueeze(-1)
                action_one_hot = torch.zeros(batch_size, self.action_dim, device=self.device)
                action_one_hot.scatter_(1, action, 1.0)
                action_one_hot = action_one_hot.unsqueeze(1)
                with torch.no_grad():
                    ensemble_next_states = [m(current_state, action_one_hot, device=self.device)[0] for m in self.world_model_ensemble]
                next_state_var = torch.var(torch.stack(ensemble_next_states, dim=0), dim=0).mean(dim=-1)
                next_state_var = torch.clamp(next_state_var, min=0.0, max=100.0)
                del ensemble_next_states
                next_state, reward_logits, continuation, hidden, _, z_t = self.world_model(current_state, action_one_hot, device=self.device)
                bins = self._symexp(torch.linspace(-20, 20, 41).to(self.device))
                reward = torch.sum(torch.softmax(reward_logits, dim=-1) * bins, dim=-1) + 0.01 * next_state_var
                reward = torch.clamp(reward, min=-10.0, max=10.0)
                imagined_states.append(next_state)
                imagined_rewards.append(reward)
                imagined_continuations.append(continuation)
                actions_taken.append(action)
                hidden_states.append(hidden)
                logger.debug(f"Imagination step {step}: hidden.shape={hidden.shape}")
                current_state = torch.cat([current_state[:, 1:, :], next_state.unsqueeze(1)], dim=1)

            bins = self._symexp(torch.linspace(-20, 20, 41).to(self.device))
            values = [torch.sum(torch.softmax(self.critic(s, h).logits, dim=-1) * bins, dim=-1) for s, h in zip(imagined_states, hidden_states)]
            values_tensor = torch.stack(values[:-1], dim=1)

            returns = []
            lambda_return = values[-1]
            for r, c in zip(reversed(imagined_rewards), reversed(imagined_continuations)):
                c = c.squeeze(-1)
                lambda_return = r + 0.985 * c * lambda_return
                returns.append(lambda_return)
            returns = torch.stack(list(reversed(returns)), dim=1)
            returns = torch.clamp(returns, min=-100.0, max=100.0)

            if self.return_range is None:
                self.return_range = torch.tensor(1.0, device=self.device)
            else:
                percentiles = torch.quantile(returns, torch.tensor([0.05, 0.95], device=self.device))
                range_estimate = percentiles[1] - percentiles[0]
                self.return_range = self.ema_alpha * self.return_range + (1 - self.ema_alpha) * range_estimate
            norm_factor = torch.max(torch.tensor(1.0, device=self.device), self.return_range)

            with torch.no_grad():
                ensemble_outputs = [m(states_tensor.unsqueeze(1), actions_tensor.unsqueeze(1), device=self.device)[0] for m in self.world_model_ensemble]
                disagreement_loss = torch.mean(torch.var(torch.stack(ensemble_outputs, dim=0)))

            self.actor_optimizer.zero_grad()
            advantages = (returns[:, :-1] - values_tensor) / norm_factor
            advantages = torch.clamp(advantages, min=-10.0, max=10.0)
            log_probs = torch.stack([self.actor(s, h).log_prob(a.squeeze(-1)) for s, h, a in zip(imagined_states[:-1], hidden_states[:-1], actions_taken[:-1])], dim=1)
            log_probs = torch.clamp(log_probs, min=-10.0, max=10.0)
            actor_loss = -(log_probs * advantages.detach()).mean()
            entropy = self.actor(states_tensor, hidden_states[0] if hidden_states else None).entropy().mean()
            total_actor_loss = actor_loss - self.entropy_coeff * entropy + 0.005 * disagreement_loss
            total_actor_loss = torch.clamp(total_actor_loss, min=-1000.0, max=1000.0)
            if self.check_training_stability(total_actor_loss):
                total_actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

            returns_detached = returns.detach()
            imagined_states_detached = [s.detach() for s in imagined_states]
            hidden_states_detached = [h.detach() for h in hidden_states]
            self.critic_optimizer.zero_grad()
            critic_logits = torch.stack([self.critic(s, h).logits for s, h in zip(imagined_states_detached[:-1], hidden_states_detached[:-1])], dim=1)
            returns_flat = returns_detached[:, :-1].reshape(-1)
            returns_flat = torch.clamp(returns_flat, min=-20.0, max=20.0)
            returns_twohot = self._twohot_encode(returns_flat, bins).view(batch_size, self.horizon - 1, 41)
            pred_probs = torch.softmax(critic_logits, dim=-1)
            critic_loss = -torch.sum(returns_twohot * torch.log(pred_probs + 1e-8), dim=-1).mean()
            critic_loss = torch.clamp(critic_loss, min=-100.0, max=100.0)
            if self.check_training_stability(critic_loss):
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

            self._update_critic_ema()

            self.training_metrics.update({
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "entropy": entropy.item(),
                "disagreement_loss": disagreement_loss.item(),
                "avg_return": returns.mean().item()
            })

            logger.info(f"{self.object_id}: Trained actor-critic with {batch_size} samples, actor loss: {actor_loss.item():.6f}, critic loss: {critic_loss.item():.6f}, avg_return: {returns.mean().item():.4f}")
        except Exception as e:
            logger.error(f"{self.object_id}: Actor-critic training error: {e}", exc_info=True)

    async def handle_state_update(self, event: Event):
        if event.origin != "ResourceManager" or not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            logger.debug(f"{self.object_id}: Invalid SYSTEM_STATE_UPDATE event: origin={event.origin}, data_type={type(event.data)}")
            return
        new_state = event.data.value
        new_ts = new_state.get("timestamp")
        if not new_ts:
            logger.warning(f"{self.object_id}: SYSTEM_STATE_UPDATE missing timestamp")
            return
        logger.debug(f"{self.object_id}: Received SYSTEM_STATE_UPDATE (ts: {new_ts:.2f})")

        if self.current_known_state is not None and TORCH_AVAILABLE:
            prev_ts = self.current_known_state.get("timestamp", 0)
            action_to_link, action_ts_to_del, latest_action_ts = None, None, -1
            history_keys = list(self.state_action_history.keys())
            for act_ts_str in history_keys:
                if act_ts_str not in self.state_action_history:
                    continue
                act_ts = float(act_ts_str)
                state_before_act, action_data = self.state_action_history[act_ts_str]
                if prev_ts <= act_ts <= new_ts or (new_ts - 60 <= act_ts <= new_ts):
                    if act_ts > latest_action_ts:
                        latest_action_ts = act_ts
                        action_to_link = action_data
                        action_ts_to_del = act_ts_str
                elif act_ts < prev_ts - 300:
                    self.state_action_history.pop(act_ts_str, None)
            if action_to_link:
                state_vec = self._vectorize_state(state_before_act)
                action_vec = self._vectorize_action(action_to_link)
                next_state_vec = self._vectorize_state(new_state)
                if state_vec is not None and action_vec is not None and next_state_vec is not None:
                    reward = self._compute_reward(state_before_act, new_state)
                    tensors = {"state_vec": state_vec, "action_vec": action_vec, "reward": reward, "next_state_vec": next_state_vec}
                    for name, tensor in tensors.items():
                        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                            logger.warning(f"{self.object_id}: Skipping transition due to NaN/Inf in {name} (shape={tensor.shape}) at ts {new_ts}")
                            break
                    else:
                        transition = (state_vec, action_vec, reward, next_state_vec)
                        if random.random() < 0.8:
                            self.replay_buffer.append(transition)
                        else:
                            self.validation_buffer.append(transition)
                        act_type = action_to_link.get("action_type", "UNKNOWN")
                        logger.info(f"{self.object_id}: Stored transition (Action '{act_type}') in replay buffer. Size: {len(self.replay_buffer)}")
                else:
                    logger.warning(f"{self.object_id}: Failed to vectorize state/action for transition at ts {new_ts}")
                if action_ts_to_del:
                    self.state_action_history.pop(action_ts_to_del, None)
            else:
                logger.debug(f"{self.object_id}: No action found between ts {prev_ts} and {new_ts} or within 60s")

        self.current_known_state = new_state

    async def handle_action_taken(self, event: Event):
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            logger.debug(f"{self.object_id}: Invalid ACTION_TAKEN event: data_type={type(event.data)}")
            return
        action_data = event.data.value
        timestamp = event.data.metadata.get("timestamp", time.time())
        if self.current_known_state is not None and TORCH_AVAILABLE:
            self.state_action_history[str(timestamp)] = (self.current_known_state, action_data)
            logger.debug(f"{self.object_id}: Stored action '{action_data.get('action_type', 'UNKNOWN')}' at ts {timestamp:.2f}. History size: {len(self.state_action_history)}")
        else:
            logger.warning(f"{self.object_id}: ACTION_TAKEN at {timestamp} but no current state or PyTorch unavailable")

    async def handle_predict_query(self, event: Event):
        if not TORCH_AVAILABLE:
            await self._send_prediction_response(event.origin, event.data.metadata.get("correlation_id"), error="PyTorch unavailable")
            return
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            await self._send_prediction_response(event.origin, event.data.metadata.get("correlation_id"), error="Invalid query format")
            return
        query_data = event.data.value
        origin_bubble_id = event.origin
        correlation_id = event.data.metadata.get("correlation_id")
        if not correlation_id:
            return
        current_state = query_data.get("current_state")
        action_to_simulate = query_data.get("action")
        if not current_state or not action_to_simulate or not isinstance(action_to_simulate, dict):
            await self._send_prediction_response(origin_bubble_id, correlation_id, error="Missing state or valid action")
            return
        act_type = action_to_simulate.get('action_type', 'UNKNOWN')
        logger.info(f"{self.object_id}: Received PREDICT_STATE_QUERY {correlation_id[:8]} from {origin_bubble_id} for action: {act_type}")

        if not self.world_model:
            await self._send_prediction_response(origin_bubble_id, correlation_id, error="DreamerV3 not available")
            return

        try:
            state_vector = self._vectorize_state(current_state)
            action_vector = self._vectorize_action(action_to_simulate)
            if state_vector is None or action_vector is None:
                raise ValueError("Vectorization failed")

            self.world_model.eval()
            with torch.no_grad():
                state_seq = state_vector.unsqueeze(0).unsqueeze(0)
                action_seq = action_vector.unsqueeze(0).unsqueeze(0)
                predicted_states, predicted_continuations = [], []
                for _ in range(self.horizon):
                    next_state, _, continuation, _, _, _ = self.world_model(state_seq, action_seq, device=self.device)
                    predicted_states.append(next_state)
                    predicted_continuations.append(continuation)
                    state_seq = torch.cat([state_seq[:, 1:, :], next_state.unsqueeze(1)], dim=1)

            predicted_state = self._devectorize_state(predicted_states[-1].squeeze(0))
            predicted_state["continuation_probability"] = predicted_continuations[-1].item()
            if "error" in predicted_state:
                raise ValueError(predicted_state["error"])

            await self._send_prediction_response(origin_bubble_id, correlation_id, prediction=predicted_state)
        except Exception as e:
            logger.error(f"{self.object_id}: Prediction error for {correlation_id[:8]}: {e}", exc_info=True)
            await self._send_prediction_response(origin_bubble_id, correlation_id, error=f"Prediction failed: {e}")

    async def _send_prediction_response(self, requester_id: Optional[str], correlation_id: Optional[str], prediction: Optional[Dict] = None, error: Optional[str] = None):
        if not requester_id or not correlation_id:
            logger.error(f"{self.object_id}: Cannot send prediction response - missing requester_id or correlation_id")
            return
        if not self.dispatcher:
            logger.error(f"{self.object_id}: Cannot send prediction response, dispatcher unavailable")
            return

        response_payload = {"correlation_id": correlation_id}
        if prediction and not error:
            response_payload["predicted_state"] = prediction
            response_payload["error"] = None
            status = "SUCCESS"
        else:
            response_payload["predicted_state"] = None
            response_payload["error"] = error if error else "Unknown prediction error"
            status = "ERROR"

        response_uc = UniversalCode(Tags.DICT, response_payload, description=f"Predicted state response ({status})")
        response_event = Event(type=Actions.PREDICT_STATE_RESPONSE, data=response_uc, origin=self.object_id, priority=2)
        await self.context.dispatch_event(response_event)
        logger.info(f"{self.object_id}: Sent PREDICT_STATE_RESPONSE ({status}) for {correlation_id[:8]} to {requester_id}")

    async def load_external_data(self, file_path: str):
        if not TORCH_AVAILABLE:
            logger.error(f"{self.object_id}: Cannot load external data, PyTorch unavailable")
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            for i, item in enumerate(data):
                state_vec = self._vectorize_state(item["state"])
                action_vec = self._vectorize_action(item["action"])
                reward = torch.tensor([item.get("reward", 0.0)], dtype=torch.float32).to(self.device)
                next_state = data[i + 1]["state"] if i + 1 < len(data) else item["state"]
                next_state_vec = self._vectorize_state(next_state)
                if state_vec is None or action_vec is None or next_state_vec is None:
                    logger.warning(f"{self.object_id}: Skipping invalid transition at index {i} due to vectorization failure")
                    continue
                tensors = {"state_vec": state_vec, "action_vec": action_vec, "reward": reward, "next_state_vec": next_state_vec}
                for name, tensor in tensors.items():
                    if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                        logger.warning(f"{self.object_id}: Skipping transition due to NaN/Inf in {name} (shape={tensor.shape}) at index {i}")
                        break
                else:
                    transition = (state_vec, action_vec, reward, next_state_vec)
                    if random.random() < 0.8:
                        self.replay_buffer.append(transition)
                    else:
                        self.validation_buffer.append(transition)
            logger.info(f"{self.object_id}: Loaded {len(self.replay_buffer)} training and {len(self.validation_buffer)} validation transitions from {file_path}")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to load external data from {file_path}: {e}", exc_info=True)

    async def add_ad_hoc_transition(self, transition_data: Dict):
        if not TORCH_AVAILABLE:
            logger.error(f"{self.object_id}: Cannot add ad hoc transition, PyTorch unavailable")
            return
        try:
            state_vec = self._vectorize_state(transition_data["state"])
            action_vec = self._vectorize_action(transition_data["action"])
            reward = torch.tensor([transition_data["reward"]], dtype=torch.float32).to(self.device)
            next_state_vec = self._vectorize_state(transition_data["next_state"])
            if state_vec is None or action_vec is None or next_state_vec is None:
                logger.error(f"{self.object_id}: Failed to vectorize ad hoc transition")
                return
            tensors = {"state_vec": state_vec, "action_vec": action_vec, "reward": reward, "next_state_vec": next_state_vec}
            for name, tensor in tensors.items():
                if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                    logger.error(f"{self.object_id}: Skipping ad hoc transition due to NaN/Inf in {name} (shape={tensor.shape})")
                    return
            transition = (state_vec, action_vec, reward, next_state_vec)
            if random.random() < 0.8:
                self.replay_buffer.append(transition)
            else:
                self.validation_buffer.append(transition)
            logger.info(f"{self.object_id}: Added ad hoc transition to replay buffer. Size: {len(self.replay_buffer)}, Validation: {len(self.validation_buffer)}")
        except Exception as e:
            logger.error(f"{self.object_id}: Error adding ad hoc transition: {e}", exc_info=True)

    async def train_world_model(self):
        if not TORCH_AVAILABLE or not self.world_model or len(self.replay_buffer) < self.batch_size:
            logger.debug(f"{self.object_id}: Skipping world model training (Torch: {TORCH_AVAILABLE}, Buffer: {len(self.replay_buffer)})")
            return
        try:
            batch_size = min(self.batch_size, len(self.replay_buffer))
            batch = random.sample(self.replay_buffer, batch_size)
            states, actions, rewards, next_states = zip(*batch)
            states_tensor = torch.stack(states).to(self.device).unsqueeze(1)
            actions_tensor = torch.stack(actions).to(self.device).unsqueeze(1)
            rewards_tensor = torch.stack(rewards).to(self.device)
            next_states_tensor = torch.stack(next_states).to(self.device)
            logger.debug(f"train_world_model: states_tensor.shape={states_tensor.shape}, actions_tensor.shape={actions_tensor.shape}")

            self.world_model.train()
            for m in self.world_model_ensemble:
                m.train()
            self.world_optimizer.zero_grad()
            for opt in self.ensemble_optimizers:
                opt.zero_grad()

            predicted_next_states, reward_logits, predicted_continuations, _, kl_loss, z_t = self.world_model(
                states_tensor, actions_tensor, device=self.device
            )

            state_loss = torch.mean((predicted_next_states - next_states_tensor) ** 2)
            bins = self._symexp(torch.linspace(-20, 20, 41).to(self.device))
            rewards_tensor = torch.clamp(rewards_tensor, min=-20.0, max=20.0)
            reward_targets = self._twohot_encode(rewards_tensor, bins)
            reward_loss = torch.nn.functional.cross_entropy(reward_logits, reward_targets, reduction='mean')
            continuation_tensor = torch.ones(batch_size, 1, dtype=torch.float32).to(self.device)
            continuation_loss = nn.BCELoss()(predicted_continuations, continuation_tensor)
            total_loss = state_loss + reward_loss + continuation_loss + 0.5 * kl_loss

            if not self.check_training_stability(total_loss):
                return

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1000)
            self.world_optimizer.step()

            for m, opt in zip(self.world_model_ensemble, self.ensemble_optimizers):
                m.train()
                opt.zero_grad()
                pred_next_states, _, _, _, _, _ = m(states_tensor, actions_tensor, device=self.device)
                ensemble_loss = torch.mean((pred_next_states - next_states_tensor) ** 2)
                ensemble_loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1000)
                opt.step()

            disagreement_loss = torch.mean(torch.var(torch.stack([m(states_tensor, actions_tensor, device=self.device)[0] for m in self.world_model_ensemble], dim=0)))

            validation_loss = await self.compute_validation_loss()

            self.training_metrics.update({
                "state_loss": state_loss.item(),
                "reward_loss": reward_loss.item(),
                "continuation_loss": continuation_loss.item(),
                "kl_loss": kl_loss.item(),
                "disagreement_loss": disagreement_loss.item(),
                "recon_loss": 0.0,
                "validation_loss": validation_loss
            })

            logger.info(f"{self.object_id}: Trained world model with {batch_size} samples, loss: {total_loss.item():.6f}, state: {state_loss.item():.6f}, reward: {reward_loss.item():.6f}, validation_loss: {validation_loss:.6f}")
        except Exception as e:
            logger.error(f"{self.object_id}: World model training error: {e}", exc_info=True)

    async def autonomous_step(self):
        await super().autonomous_step()
        self.execution_count += 1
        if self.execution_count % 10 == 0:
            self.collect_transitions(10)
            logger.info(f"{self.object_id}: Attempting training, replay buffer size: {len(self.replay_buffer)}")
            await self.train_world_model()
            await self.train_actor_critic()
        await asyncio.sleep(0.5)
