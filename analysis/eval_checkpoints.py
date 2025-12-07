"""
Lightweight evaluation of the saved SAC checkpoints using a synthetic lane-following
environment. The real DonkeySim/robot environment is not available in this context,
so this script uses a reproducible proxy task to compare the checkpoints and to
generate summary curves.

Improvements in this rewritten version:
- Added trajectory logging and optional plotting for debugging crashes.
- Added evaluation of a baseline (random or hand-crafted) policy for comparison.
- Enhanced metrics: Added mean absolute position/heading deviation.
- Added progress bar for checkpoint evaluation using tqdm.
- Added fallback defaults for config values if config.yml is missing or incomplete.
- Improved error handling: Skip corrupt checkpoints and log warnings.
- Parallelized evaluation using multiprocessing for faster processing on multi-core systems.
- Tuned env: Made track slightly wider (abs(pos) > 3.0) for easier solvability in proxy.
- Added action statistics (mean/std steering/throttle) to EvalResult.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend for plot export
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gym import Env, spaces
from stable_baselines3 import SAC
from tqdm import tqdm
import yaml  # Added for direct yaml loading

import cloudpickle.cloudpickle as _cp

def _make_cell(value=None):
    def inner():
        return value
    return inner.__closure__[0]

# Patch for cloudpickle compatibility
if not hasattr(_cp, "_make_cell"):
    _cp._make_cell = _make_cell

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# from learning_racer.config import ConfigReader  # Comment out if not needed; using yaml directly

@dataclass
class EvalResult:
    steps: int
    mean_reward: float
    reward_std: float
    success_rate: float
    mean_length: float
    mean_abs_pos: float  # New: Mean absolute position deviation
    mean_abs_heading: float  # New: Mean absolute heading deviation
    mean_steer: float  # New: Action stats
    std_steer: float
    mean_throttle: float
    std_throttle: float
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

    def __init__(self, max_steps: int = 200, seed: int = 0, config_path: Optional[str] = "config.yml"):
        super().__init__()
        self.config = self._load_config(config_path)

        self.z_dim = self.config.get('sac_variants_size', 32)  # Fallback default
        self.n_history = self.config.get('agent_n_command_history', 3)
        self.min_steer = self.config.get('agent_min_steering', -1.0)
        self.max_steer = self.config.get('agent_max_steering', 1.0)
        self.min_throttle = self.config.get('agent_min_throttle', 0.0)
        self.max_throttle = self.config.get('agent_max_throttle', 1.0)

        # Debug print to check loaded values
        print(f"Loaded config values: z_dim={self.z_dim}, n_history={self.n_history}, obs_shape={self.z_dim + 2 * self.n_history}")

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
        self.track_width = 3.0  # Tuned: Slightly wider track for proxy solvability

        self.pos = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.t = 0
        self.action_hist: List[float] = [0.0 for _ in range(self.n_history * 2)]
        self.reset()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        config = {}
        try:
            # Support relative path by resolving from current working directory or script dir
            config_path = Path(config_path).resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            # Extract nested keys with correct casing
            sac_setting = yaml_data.get('SAC_SETTING', {})
            agent_setting = yaml_data.get('AGENT_SETTING', {})
            config['sac_variants_size'] = sac_setting.get('VARIANTS_SIZE', 32)
            config['agent_n_command_history'] = agent_setting.get('N_COMMAND_HISTORY', 3)
            config['agent_min_steering'] = agent_setting.get('MIN_STEERING', -1.0)
            config['agent_max_steering'] = agent_setting.get('MAX_STEERING', 1.0)
            config['agent_min_throttle'] = agent_setting.get('MIN_THROTTLE', 0.0)
            config['agent_max_throttle'] = agent_setting.get('MAX_THROTTLE', 1.0)
            print(f"Successfully loaded config from {config_path}: {config}")
            return config
        except Exception as e:
            warnings.warn(f"Config loading failed: {e}. Using defaults: {config}")
            return config

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
        # Simple kinematic update with a small amount of noise
        self.heading += 0.06 * steer + 0.02 * self.heading + float(self.np_random.normal(scale=0.01))
        self.speed = 0.85 * self.speed + 0.15 * throttle
        self.pos += 0.05 * self.speed * math.sin(self.heading) + 0.02 * steer
        self.t += 1

        self._record_action(scaled_action.tolist())

        off_track = abs(self.pos) > self.track_width
        done = bool(off_track or self.t >= self.max_steps)
        reward = self._reward(scaled_action, off_track)
        obs = self._get_obs()
        info = {
            "off_track": off_track,
            "steps": self.t,
            "pos": self.pos,
            "heading": self.heading,
            "speed": self.speed,
            "action": scaled_action.tolist(),  # For action stats
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
        smooth_penalty = 0.02 * np.std(self.action_hist[-6:]) if len(self.action_hist) >= 6 else 0.0
        speed_reward = 0.5 * (throttle - self.min_throttle) / max(self.max_throttle - self.min_throttle, 1e-6)
        reward = 1.2 - center_penalty - curvature_penalty - smooth_penalty + speed_reward
        if off_track:
            reward -= 5.0
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        latent = np.zeros(self.z_dim, dtype=np.float32)
        latent[0] = np.tanh(self.pos / self.track_width)
        latent[1] = np.tanh(self.heading)
        latent[2] = np.clip(self.speed / 1.5, 0.0, 1.0)
        latent[3] = np.mean(self.action_hist[-4:]) if len(self.action_hist) >= 4 else 0.0
        latent += self.np_random.normal(scale=0.02, size=self.z_dim)
        history = np.asarray(self.action_hist, dtype=np.float32)
        return np.concatenate([latent, history]).astype(np.float32)


def evaluate_checkpoint(
    model_path: Path, episodes: int = 15, max_steps: int = 200, seed: int = 42, config_path: Optional[str] = "config.yml",
    baseline_policy: Optional[str] = None, plot_trajectories: bool = False, out_dir: Optional[Path] = None
) -> EvalResult:
    env = SyntheticTrackEnv(max_steps=max_steps, seed=seed, config_path=config_path)
    model = None
    if not baseline_policy:
        try:
            model = SAC.load(str(model_path), env=env, device="cpu")
        except Exception as e:
            warnings.warn(f"Failed to load {model_path}: {e}. Skipping.")
            return None

    rewards, lengths, successes = [], [], []
    abs_pos_list, abs_heading_list = [], []
    steers, throttles = [], []
    traj_infos = [] if plot_trajectories else None

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        ep_infos = [] if plot_trajectories else None
        ep_abs_pos, ep_abs_heading = [], []
        ep_steers, ep_throttles = [], []

        while not done:
            if baseline_policy == "random":
                action = env.action_space.sample()
            elif baseline_policy == "straight":
                action = np.array([0.0, 1.0])  # Center steer, max throttle
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            if plot_trajectories:
                ep_infos.append(info)
            ep_abs_pos.append(abs(info["pos"]))
            ep_abs_heading.append(abs(info["heading"]))
            ep_steers.append(info["action"][0])
            ep_throttles.append(info["action"][1])

        rewards.append(ep_reward)
        lengths.append(steps)
        successes.append(0.0 if info.get("off_track", False) else 1.0)
        abs_pos_list.append(np.mean(ep_abs_pos))
        abs_heading_list.append(np.mean(ep_abs_heading))
        steers.extend(ep_steers)
        throttles.extend(ep_throttles)
        if plot_trajectories:
            traj_infos.append(ep_infos)

    if plot_trajectories and out_dir:
        plot_trajectories_func(traj_infos, out_dir / f"trajectories_{model_path.stem}.png")

    # Proxy losses and entropy
    actor_loss, critic_loss = proxy_losses(
        model if model else None, SyntheticTrackEnv(max_steps=max_steps, seed=seed + 123, config_path=config_path), baseline_policy
    )
    entropy = policy_entropy(model if model else None, env, baseline_policy=baseline_policy)

    step_count = extract_step_count(model_path) if not baseline_policy else 0
    return EvalResult(
        steps=step_count,
        mean_reward=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)),
        success_rate=float(np.mean(successes)),
        mean_length=float(np.mean(lengths)),
        mean_abs_pos=float(np.mean(abs_pos_list)),
        mean_abs_heading=float(np.mean(abs_heading_list)),
        mean_steer=float(np.mean(steers)),
        std_steer=float(np.std(steers)),
        mean_throttle=float(np.mean(throttles)),
        std_throttle=float(np.std(throttles)),
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy=entropy,
    )


def plot_trajectories_func(traj_infos: List[List[Dict]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, traj in enumerate(traj_infos):
        pos = [info['pos'] for info in traj]
        ax.plot(pos, label=f'Episode {i+1}')
    ax.axhline(3.0, color='r', linestyle='--', label='Track Limit')
    ax.axhline(-3.0, color='r', linestyle='--')
    ax.set_title("Episode Trajectories (Position over Steps)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Position")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def gather_rollout(
    model: Optional[SAC], env: SyntheticTrackEnv, batch_size: int = 512, baseline_policy: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_buf, action_buf, reward_buf, next_obs_buf, done_buf = [], [], [], [], []
    obs = env.reset()
    for _ in range(batch_size):
        if baseline_policy == "random":
            action = env.action_space.sample()
        elif baseline_policy == "straight":
            action = np.array([0.0, 1.0])
        else:
            action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, done, _ = env.step(action)
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


def proxy_losses(model: Optional[SAC], env: SyntheticTrackEnv, baseline_policy: Optional[str] = None) -> Tuple[float, float]:
    if baseline_policy:
        return 0.0, 0.0  # No losses for baselines
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


def policy_entropy(model: Optional[SAC], env: SyntheticTrackEnv, n_samples: int = 256, baseline_policy: Optional[str] = None) -> float:
    if baseline_policy:
        return 0.0  # No entropy for baselines
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
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # Expanded for new metrics

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

    axes[2, 0].plot(df["steps"], df["mean_abs_pos"], label="Mean |pos|")
    axes[2, 0].plot(df["steps"], df["mean_abs_heading"], label="Mean |heading|")
    axes[2, 0].set_title("Stability Metrics")
    axes[2, 0].set_xlabel("Training steps")
    axes[2, 0].legend()

    axes[2, 1].plot(df["steps"], df["mean_steer"], label="Mean steer")
    axes[2, 1].fill_between(df["steps"], df["mean_steer"] - df["std_steer"], df["mean_steer"] + df["std_steer"], alpha=0.2)
    axes[2, 1].plot(df["steps"], df["mean_throttle"], label="Mean throttle")
    axes[2, 1].fill_between(df["steps"], df["mean_throttle"] - df["std_throttle"], df["mean_throttle"] + df["std_throttle"], alpha=0.2)
    axes[2, 1].set_title("Action Statistics")
    axes[2, 1].set_xlabel("Training steps")
    axes[2, 1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAC checkpoints with a synthetic lane-following proxy (aligned with docker/simulator/sim.sh defaults)."
    )
    parser.add_argument("--log-dir", default="model_log", help="Directory containing model_*_steps.zip checkpoints.")
    parser.add_argument("--config", default="config.yml", help="Path to config file (supports relative paths, mirrors sim.sh --config).")
    parser.add_argument("--out-dir", default="analysis/plots", help="Where to write plots/CSV.")
    parser.add_argument("--episodes", type=int, default=15, help="Episodes per checkpoint.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per synthetic episode.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--pattern",
        default="model_*_steps.zip",
        help="Glob for checkpoints inside log-dir (matches sim.sh naming).",
    )
    parser.add_argument("--plot-trajectories", action="store_true", help="Plot episode trajectories for each checkpoint.")
    parser.add_argument("--eval-baseline", choices=["random", "straight"], default=None, help="Evaluate a baseline policy instead of models.")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count() // 2, help="Number of parallel workers for evaluation.")
    return parser.parse_args()


def collect_checkpoints(model_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(model_dir.glob(pattern), key=extract_step_count)


def eval_worker(args_tuple):
    model_path, args = args_tuple
    return evaluate_checkpoint(
        model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        config_path=args.config,
        baseline_policy=args.eval_baseline,
        plot_trajectories=args.plot_trajectories,
        out_dir=Path(args.out_dir),
    )


def main():
    args = parse_args()
    model_dir = Path(args.log_dir)
    checkpoints = list(collect_checkpoints(model_dir, args.pattern))
    if not checkpoints and not args.eval_baseline:
        raise SystemExit(f"No checkpoints found in {model_dir} matching {args.pattern}")

    results: List[EvalResult] = []

    if args.eval_baseline:
        # Single baseline eval
        print(f"Evaluating baseline policy: {args.eval_baseline}")
        baseline_result = evaluate_checkpoint(
            Path("dummy"),  # Dummy path for baseline
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            config_path=args.config,
            baseline_policy=args.eval_baseline,
            plot_trajectories=args.plot_trajectories,
            out_dir=Path(args.out_dir),
        )
        if baseline_result:
            results.append(baseline_result)
    else:
        # Parallel eval for checkpoints
        with mp.Pool(args.num_workers) as pool:
            worker_args = [(ckpt, args) for ckpt in checkpoints]
            for result in tqdm(pool.imap_unordered(eval_worker, worker_args), total=len(checkpoints), desc="Evaluating checkpoints"):
                if result:
                    results.append(result)

    if not results:
        raise SystemExit("No valid results generated.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(results, out_dir / "checkpoint_curves.png")

    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("steps")
    df.to_csv(out_dir / "checkpoint_metrics.csv", index=False)
    print(df)
    if not args.eval_baseline:
        best_idx = df["mean_reward"].idxmax()
        print(f"Best checkpoint: {df.loc[best_idx, 'steps']} steps (mean reward={df.loc[best_idx, 'mean_reward']:.2f})")


if __name__ == "__main__":
    main()
