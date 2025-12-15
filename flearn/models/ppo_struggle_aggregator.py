"""Struggle-aware contextual PPO aggregator.

This module introduces a PPO variant where client struggle scores directly
influence the predicted aggregation weights. It does so by blending the policy
output with a temperature-scaled distribution over the normalized struggle
scores, ensuring that struggling clients receive more emphasis unless the
policy decisively overrides them.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .ppo_aggregator import (
    AggregationEnvPPO,
    PPOAgentR,
)


class StruggleAwareAggregationEnv(AggregationEnvPPO):
    """Environment that injects struggle-driven attention into weight selection."""

    def __init__(
        self,
        *args,
        struggle_blend: float = 0.35,
        struggle_temperature: float = 2.0,
        struggle_floor: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.struggle_blend = float(max(0.0, min(1.0, struggle_blend)))
        self.struggle_temperature = float(max(0.1, struggle_temperature))
        self.struggle_floor = float(max(1e-6, struggle_floor))

    def _struggle_attention(self, client_ids: List) -> torch.Tensor:
        if not client_ids:
            return torch.ones(1, device=self.device)
        scores = torch.tensor(
            [self.client_struggle_scores.get(cid, 0.0) for cid in client_ids],
            dtype=torch.float32,
            device=self.device,
        )
        if torch.allclose(scores, scores[0]):
            attn = torch.ones_like(scores)
        else:
            centered = scores - scores.mean()
            attn = torch.softmax(self.struggle_temperature * centered, dim=0)
        attn = attn.clamp(min=self.struggle_floor)
        return attn / attn.sum()

    def _blend_with_baseline(
        self,
        action: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        adjusted = super()._blend_with_baseline(action, baseline)
        if self.struggle_blend <= 1e-6:
            return adjusted
        client_ids = (
            list(self.weights_vector.keys())
            if isinstance(self.weights_vector, dict)
            else list(range(len(adjusted)))
        )
        struggle_attn = self._struggle_attention(client_ids)
        blended = (1 - self.struggle_blend) * adjusted + self.struggle_blend * struggle_attn
        blended = blended.clamp(min=1e-6)
        return blended / blended.sum()


class StruggleAwarePPOAgent(PPOAgentR):
    """PPO agent that operates with the struggle-aware aggregation environment."""

    def __init__(
        self,
        *,
        eval_fn,
        build_class_prototypes,
        num_classes: int,
        device,
        num_clients_per_round: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        baseline_type: str = "mean",
        reward_case: str = "acc_align",
        use_advantage: bool = True,
        kl_weight: float = 0.01,
        delta_scale: float = 0.5,
        adaptive_delta: bool = True,
        delta_scale_min: float = 0.1,
        delta_scale_max: float = 0.9,
        delta_scale_gain: float = 0.05,
        delta_scale_margin: float = 0.002,
        history_size: int = 5,
        context_variant: str = "enhanced",
        enable_safety: bool = True,
        safety_tolerance: float = 0.005,
        safety_patience: int = 2,
        safety_cooldown_rounds: int = 1,
        safety_penalty: float = 0.05,
        struggle_blend: float = 0.35,
        struggle_temperature: float = 2.0,
        struggle_floor: float = 1e-4,
        **kwargs,
    ):
        super().__init__(
            eval_fn=eval_fn,
            build_class_prototypes=build_class_prototypes,
            num_classes=num_classes,
            device=device,
            num_clients_per_round=num_clients_per_round,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            baseline_type=baseline_type,
            reward_case=reward_case,
            use_advantage=use_advantage,
            kl_weight=kl_weight,
            delta_scale=delta_scale,
            adaptive_delta=adaptive_delta,
            delta_scale_min=delta_scale_min,
            delta_scale_max=delta_scale_max,
            delta_scale_gain=delta_scale_gain,
            delta_scale_margin=delta_scale_margin,
            history_size=history_size,
            context_variant=context_variant,
            enable_safety=enable_safety,
            safety_tolerance=safety_tolerance,
            safety_patience=safety_patience,
            safety_cooldown_rounds=safety_cooldown_rounds,
            safety_penalty=safety_penalty,
            **kwargs,
        )
        self.env = StruggleAwareAggregationEnv(
            eval_fn,
            build_class_prototypes,
            num_classes,
            num_clients_per_round,
            device,
            baseline_type=baseline_type,
            reward_case=reward_case,
            use_advantage=use_advantage,
            kl_weight=kl_weight,
            verbose=getattr(self.env, "verbose", False),
            delta_scale=delta_scale,
            adaptive_delta=adaptive_delta,
            delta_scale_min=delta_scale_min,
            delta_scale_max=delta_scale_max,
            delta_scale_gain=delta_scale_gain,
            delta_scale_margin=delta_scale_margin,
            history_size=history_size,
            context_variant=context_variant,
            enable_safety=enable_safety,
            safety_tolerance=safety_tolerance,
            safety_patience=safety_patience,
            safety_cooldown_rounds=safety_cooldown_rounds,
            safety_penalty=safety_penalty,
            struggle_blend=struggle_blend,
            struggle_temperature=struggle_temperature,
            struggle_floor=struggle_floor,
        )
