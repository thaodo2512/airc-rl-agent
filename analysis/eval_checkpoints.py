"""
Lightweight evaluation of the saved SAC checkpoints using a synthetic lane-following
environment. The real DonkeySim/robot environment is not available in this context,
so this script uses a reproducible proxy task to compare the checkpoints and to
generate summary curves.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import matplotlib
import sys

matplotlib.use("Agg")  # headless backend for plot export
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gym import Env, spaces
from stable_baselines3 import SAC

import cloudpickle.cloudpickle as _cp

def _make_cell(value=None):
    def inner():
        return value
    return inner.__closure__[0]

# Some checkpoints were saved with a newer cloudpickle; patch the helper to stay backward compatible.
if not hasattr(_cp, "_make_cell"):
    _cp._make_cell = _make_cell

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning_racer.config import ConfigReader


@dataclass
class EvalResult:
    steps: int
    mean_reward: float
    reward_std: float
    success_rate: float
    mean_length: float
    actor_loss: float
    critic_loss: float
    entropy: float


class SyntheticTrackEnv(Env):
    """
    A small deterministic surrogate for the JetBot/DonkeyCar task.
    Observation: concatenation of a latent vector (z_dim) and the last N actions.
    Action: [steering, throttle] in [-1, 1] x [-1, 1] (throttle is rescaled internally).
    """

    metadata = {"render.modes": []}

    def __init__(self, max_steps: int = 200, seed: int = 0, config_path: str = "config.yml"):
        super().__init__()
        config = ConfigReader()
        config.load(config_path)

        self.z_dim = config.sac_variants_size()
        self.n_history = config.agent_n_command_history()
        self.min_steer = config.agent_min_steering()
        self.max_steer = config.agent_max_steering()
        self.min_throttle = config.agent_min_throttle()
        self.max_throttle = config.agent_max_throttle()

        self.action_space = spaces.Box(
            low=np.array([self.min_steer, -1], dtype=np.float32),
            high=np.array([self.max_steer, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self.z_dim + 2 * self.n_history,),
            dtype=np.float32,
        )
        self.max_steps = max_steps
        self.np_random = np.random.RandomState(seed)

        self.pos = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.t = 0
        self.action_hist: List[float] = [0.0 for _ in range(self.n_history * 2)]
        self.reset()

    def seed(self, seed: int = None) -> List[int]:
        self.np_random.seed(seed)
        return [seed]

    def reset(self):
        self.pos = float(self.np_random.uniform(-0.5, 0.5))
        self.heading = float(self.np_random.uniform(-0.05, 0.05))
        self.speed = float(self.np_random.uniform(0.7, 0.9))
        self.t = 0
        self.action_hist = [0.0 for _ in range(self.n_history * 2)]
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        scaled_action = self._scale_action(np.asarray(action, dtype=np.float32))
        steer, throttle = scaled_action
        # Simple kinematic update with a small amount of noise to avoid overfitting
        self.heading += 0.06 * steer + 0.02 * self.heading + float(self.np_random.normal(scale=0.01))
        self.speed = 0.85 * self.speed + 0.15 * throttle
        self.pos += 0.05 * self.speed * math.sin(self.heading) + 0.02 * steer
        self.t += 1

        self._record_action(scaled_action.tolist())

        off_track = abs(self.pos) > 2.5
        done = bool(off_track or self.t >= self.max_steps)
        reward = self._reward(scaled_action, off_track)
        obs = self._get_obs()
        info = {
            "off_track": off_track,
            "steps": self.t,
            "pos": self.pos,
            "heading": self.heading,
            "speed": self.speed,
        }
        return obs, reward, done, info

    def render(self, mode="human"):
        return None

    def close(self):
        return None

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        t = (action[1] + 1.0) / 2.0
        throttle = (1 - t) * self.min_throttle + self.max_throttle * t
        return np.asarray([action[0], throttle], dtype=np.float32)

    def _record_action(self, scaled_action: List[float]) -> None:
        if len(self.action_hist) >= self.n_history * 2:
            self.action_hist = self.action_hist[2:]
        self.action_hist.extend(scaled_action)

    def _reward(self, scaled_action: np.ndarray, off_track: bool) -> float:
        steer, throttle = scaled_action
        center_penalty = 0.6 * abs(self.pos)
        curvature_penalty = 0.2 * abs(self.heading)
        smooth_penalty = 0.02 * np.std(self.action_hist[-6:]) if self.action_hist else 0.0
        speed_reward = 0.5 * (throttle - self.min_throttle) / max(self.max_throttle - self.min_throttle, 1e-6)
        reward = 1.2 - center_penalty - curvature_penalty - smooth_penalty + speed_reward
        if off_track:
            reward -= 5.0
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        latent = np.zeros(self.z_dim, dtype=np.float32)
        latent[0] = np.tanh(self.pos / 2.5)
        latent[1] = np.tanh(self.heading)
        latent[2] = np.clip(self.speed / 1.5, 0.0, 1.0)
        latent[3] = np.mean(self.action_hist[-4:]) if self.action_hist else 0.0
        latent += self.np_random.normal(scale=0.02, size=self.z_dim)
        history = np.asarray(self.action_hist, dtype=np.float32)
        return np.concatenate([latent, history]).astype(np.float32)


def evaluate_checkpoint(
    model_path: Path, episodes: int = 15, max_steps: int = 200, seed: int = 42, config_path: str = "config.yml"
) -> EvalResult:
    env = SyntheticTrackEnv(max_steps=max_steps, seed=seed, config_path=config_path)
    model = SAC.load(str(model_path), env=env, device="cpu")

    rewards, lengths, successes = [], [], []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        last_info: Dict = {}
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            last_info = info
        rewards.append(ep_reward)
        lengths.append(steps)
        successes.append(0.0 if last_info.get("off_track", False) else 1.0)

    # Proxy losses computed on fresh rollouts so we get comparable magnitude curves
    actor_loss, critic_loss = proxy_losses(
        model, SyntheticTrackEnv(max_steps=max_steps, seed=seed + 123, config_path=config_path)
    )
    entropy = policy_entropy(model, env)

    step_count = extract_step_count(model_path)
    return EvalResult(
        steps=step_count,
        mean_reward=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)),
        success_rate=float(np.mean(successes)),
        mean_length=float(np.mean(lengths)),
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy=entropy,
    )


def gather_rollout(
    model: SAC, env: SyntheticTrackEnv, batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_buf, action_buf, reward_buf, next_obs_buf, done_buf = [], [], [], [], []
    obs = env.reset()
    for _ in range(batch_size):
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        obs_buf.append(obs)
        action_buf.append(action)
        reward_buf.append(reward)
        next_obs_buf.append(next_obs)
        done_buf.append(float(done))
        obs = next_obs if not done else env.reset()
    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(action_buf, dtype=np.float32),
        np.asarray(reward_buf, dtype=np.float32),
        np.asarray(next_obs_buf, dtype=np.float32),
        np.asarray(done_buf, dtype=np.float32),
    )


def proxy_losses(model: SAC, env: SyntheticTrackEnv) -> Tuple[float, float]:
    obs, actions, rewards, next_obs, dones = gather_rollout(model, env)
    device = model.device
    obs_t = torch.as_tensor(obs, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    rewards_t = torch.as_tensor(rewards, device=device).unsqueeze(-1)
    next_obs_t = torch.as_tensor(next_obs, device=device)
    dones_t = torch.as_tensor(dones, device=device).unsqueeze(-1)

    with torch.no_grad():
        next_actions, next_log_prob = model.actor.action_log_prob(next_obs_t)
        next_q_values = torch.cat(model.critic_target(next_obs_t, next_actions), dim=1)
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        ent_coef = torch.exp(model.log_ent_coef.detach()) if model.ent_coef_optimizer is not None else model.ent_coef_tensor
        target_q_values = rewards_t + (1 - dones_t) * model.gamma * (next_q_values - ent_coef * next_log_prob.unsqueeze(-1))

    current_q_values = model.critic(obs_t, actions_t)
    critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

    actions_pi, log_prob = model.actor.action_log_prob(obs_t)
    q_values_pi = torch.cat(model.critic(obs_t, actions_pi), dim=1)
    min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
    actor_loss = (ent_coef * log_prob.unsqueeze(-1) - min_qf_pi).mean()

    return float(actor_loss.detach().cpu().item()), float(critic_loss.detach().cpu().item())


def policy_entropy(model: SAC, env: SyntheticTrackEnv, n_samples: int = 256) -> float:
    """
    Monte-Carlo estimate of the policy entropy on rollouts gathered from the proxy env.
    Compatible with both diagonal Gaussian and gSDE policies.
    """
    obs_buffer = []
    obs = env.reset()
    for _ in range(n_samples):
        action, _ = model.predict(obs, deterministic=False)
        obs_buffer.append(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    obs_t = torch.as_tensor(np.asarray(obs_buffer, dtype=np.float32), device=model.device)
    with torch.no_grad():
        _, log_prob = model.actor.action_log_prob(obs_t)
        entropy_est = (-log_prob).mean().item()
    return float(entropy_est)


def extract_step_count(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


def plot_curves(results: List[EvalResult], output_path: Path) -> None:
    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("steps")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(df["steps"], df["mean_reward"], label="Mean reward")
    axes[0, 0].fill_between(
        df["steps"], df["mean_reward"] - df["reward_std"], df["mean_reward"] + df["reward_std"], alpha=0.2
    )
    axes[0, 0].set_title("Evaluation reward")
    axes[0, 0].set_xlabel("Training steps (checkpoint)")
    axes[0, 0].set_ylabel("Episode reward")

    axes[0, 1].plot(df["steps"], df["actor_loss"], label="Actor loss")
    axes[0, 1].plot(df["steps"], df["critic_loss"], label="Critic loss")
    axes[0, 1].set_title("Proxy losses")
    axes[0, 1].set_xlabel("Training steps")
    axes[0, 1].legend()

    axes[1, 0].plot(df["steps"], df["entropy"], color="tab:green", label="Policy entropy")
    axes[1, 0].set_title("Policy entropy")
    axes[1, 0].set_xlabel("Training steps")
    axes[1, 0].set_ylabel("Entropy (nats)")

    axes[1, 1].plot(df["steps"], df["success_rate"], color="tab:purple", label="Success rate")
    axes[1, 1].set_title("Success rate (no crash)")
    axes[1, 1].set_xlabel("Training steps")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_ylim(0, 1.05)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAC checkpoints with a synthetic lane-following proxy (aligned with docker/simulator/sim.sh defaults)."
    )
    parser.add_argument("--log-dir", default="model_log", help="Directory containing model_*_steps.zip checkpoints.")
    parser.add_argument("--config", default="config.yml", help="Config file (mirrors sim.sh --config).")
    parser.add_argument("--out-dir", default="analysis/plots", help="Where to write plots/CSV.")
    parser.add_argument("--episodes", type=int, default=15, help="Episodes per checkpoint.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per synthetic episode.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--pattern",
        default="model_*_steps.zip",
        help="Glob for checkpoints inside log-dir (matches sim.sh naming).",
    )
    return parser.parse_args()


def collect_checkpoints(model_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(model_dir.glob(pattern), key=extract_step_count)


def main():
    args = parse_args()
    model_dir = Path(args.log_dir)
    checkpoints = list(collect_checkpoints(model_dir, args.pattern))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {model_dir} matching {args.pattern}")

    results: List[EvalResult] = []

    for ckpt in checkpoints:
        print(f"Evaluating {ckpt} ...")
        results.append(
            evaluate_checkpoint(
                ckpt,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                config_path=args.config,
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(results, out_dir / "checkpoint_curves.png")

    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("steps")
    df.to_csv(out_dir / "checkpoint_metrics.csv", index=False)
    print(df)
    best_idx = df["mean_reward"].idxmax()
    print(f"Best checkpoint: {df.loc[best_idx, 'steps']} steps (mean reward={df.loc[best_idx, 'mean_reward']:.2f})")


if __name__ == "__main__":
    main()
