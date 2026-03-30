"""Generic tabular Q-learning trainer/evaluator with TensorBoard + GIF export.

Works for discrete-state/discrete-action Gymnasium environments
(example: FrozenLake-v1, Taxi-v3).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


DEFAULT_ENV_ID = "FrozenLake-v1"


def env_slug(env_id: str) -> str:
    """Create filesystem-safe slug from env id."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", env_id).strip("_").lower()


def parse_env_kwargs(raw: str) -> dict[str, Any]:
    """Parse JSON env kwargs passed from CLI."""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --env-kwargs JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--env-kwargs must be a JSON object, e.g. '{\"map_name\": \"8x8\"}'.")
    return data


def normalize_model_path(path: Path) -> Path:
    """Ensure model path has .npy extension."""
    return path if path.suffix == ".npy" else path.with_suffix(".npy")


def build_env(
    env_id: str,
    env_kwargs: dict[str, Any],
    render_mode: str | None = None,
) -> gym.Env:
    """Create a discrete Gymnasium environment."""
    kwargs = dict(env_kwargs)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise TypeError(
            f"Unsupported observation space for tabular Q-learning: {env.observation_space}. "
            "Need gym.spaces.Discrete."
        )
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError(
            f"Unsupported action space for tabular Q-learning: {env.action_space}. "
            "Need gym.spaces.Discrete."
        )
    return env


def initialize_q_table(state_space: int, action_space: int) -> np.ndarray:
    return np.zeros((state_space, action_space), dtype=np.float32)


def greedy_policy(q_table: np.ndarray, state: int) -> int:
    return int(np.argmax(q_table[state]))


def epsilon_greedy_policy(
    q_table: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
    action_space: int,
) -> int:
    if rng.random() > epsilon:
        return greedy_policy(q_table, state)
    return int(rng.integers(0, action_space))


def evaluate_agent(
    env: gym.Env,
    q_table: np.ndarray,
    max_steps: int,
    episodes: int,
    seed: int,
    success_threshold: float,
) -> tuple[float, float, float, float]:
    """Evaluate greedy policy and return mean/std reward, success rate, mean episode length."""
    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(1, max_steps + 1):
            action = greedy_policy(q_table, int(state))
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            state = new_state

            if terminated or truncated:
                lengths.append(step)
                break
        else:
            # Runs only if loop did NOT break (episode hit max_steps exactly).
            lengths.append(max_steps)

        rewards.append(total_reward)
        successes.append(1.0 if total_reward >= success_threshold else 0.0)

    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        float(np.mean(successes)),
        float(np.mean(lengths)),
    )


def save_q_table(q_table: np.ndarray, model_path: Path, metadata: dict[str, Any] | None = None) -> None:
    model_path = normalize_model_path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(model_path, q_table)
    if metadata is not None:
        model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_q_table(model_path: Path) -> np.ndarray:
    model_path = normalize_model_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Q-table not found at: {model_path}. Train first with --train.")
    q_table = np.load(model_path)
    if q_table.ndim != 2:
        raise ValueError(f"Invalid Q-table shape: {q_table.shape}")
    return q_table.astype(np.float32)


def validate_q_table_shape(q_table: np.ndarray, env: gym.Env) -> None:
    expected_shape = (env.observation_space.n, env.action_space.n)
    if q_table.shape != expected_shape:
        raise ValueError(
            f"Q-table shape {q_table.shape} does not match env shape {expected_shape}. "
            "Use the correct --env-id/--env-kwargs or retrain the table."
        )


def train(
    env: gym.Env,
    eval_env: gym.Env,
    env_id: str,
    env_kwargs: dict[str, Any],
    episodes: int,
    learning_rate: float,
    gamma: float,
    min_epsilon: float,
    max_epsilon: float,
    decay_rate: float,
    max_steps: int,
    model_path: Path,
    log_dir: Path,
    eval_freq: int,
    checkpoint_freq: int,
    eval_episodes: int,
    seed: int,
    success_threshold: float,
) -> np.ndarray:
    """Train Q-learning agent with TensorBoard tracking."""
    state_space = env.observation_space.n
    action_space = env.action_space.n
    q_table = initialize_q_table(state_space, action_space)

    model_path = normalize_model_path(model_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    rng = np.random.default_rng(seed)

    reward_window: list[float] = []
    success_window: list[float] = []
    best_eval_reward = -np.inf

    pbar = tqdm(range(1, episodes + 1), desc=f"Training Q-learning ({env_id})")
    for episode in pbar:
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * (episode - 1))

        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        td_errors: list[float] = []

        for step in range(1, max_steps + 1):
            action = epsilon_greedy_policy(q_table, int(state), epsilon, rng, action_space)
            new_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            best_next_q = 0.0 if done else float(np.max(q_table[int(new_state)]))
            td_target = float(reward) + gamma * best_next_q
            td_error = td_target - float(q_table[int(state), action])
            q_table[int(state), action] += learning_rate * td_error

            td_errors.append(abs(td_error))
            state = new_state
            total_reward += float(reward)

            if done:
                break
        else:
            step = max_steps

        success = 1.0 if total_reward >= success_threshold else 0.0
        reward_window.append(total_reward)
        success_window.append(success)
        if len(reward_window) > 100:
            reward_window.pop(0)
        if len(success_window) > 100:
            success_window.pop(0)

        avg_reward_100 = float(np.mean(reward_window))
        avg_success_100 = float(np.mean(success_window))
        mean_td_error = float(np.mean(td_errors)) if td_errors else 0.0

        writer.add_scalar("train/episode_reward", total_reward, episode)
        writer.add_scalar("train/episode_length", step, episode)
        writer.add_scalar("train/epsilon", epsilon, episode)
        writer.add_scalar("train/td_error_mean", mean_td_error, episode)
        writer.add_scalar("train/success", success, episode)
        writer.add_scalar("train/reward_mean_100", avg_reward_100, episode)
        writer.add_scalar("train/success_rate_100", avg_success_100, episode)
        writer.add_scalar("train/q_value_mean", float(np.mean(q_table)), episode)
        writer.add_scalar("train/q_value_max", float(np.max(q_table)), episode)

        pbar.set_postfix(
            reward=f"{total_reward:.2f}",
            eps=f"{epsilon:.3f}",
            sr100=f"{avg_success_100:.3f}",
        )

        if eval_freq > 0 and episode % eval_freq == 0:
            mean_reward, std_reward, success_rate, mean_ep_len = evaluate_agent(
                env=eval_env,
                q_table=q_table,
                max_steps=max_steps,
                episodes=eval_episodes,
                seed=seed + 10_000,
                success_threshold=success_threshold,
            )
            writer.add_scalar("eval/mean_reward", mean_reward, episode)
            writer.add_scalar("eval/std_reward", std_reward, episode)
            writer.add_scalar("eval/success_rate", success_rate, episode)
            writer.add_scalar("eval/mean_episode_length", mean_ep_len, episode)

            if mean_reward > best_eval_reward:
                best_eval_reward = mean_reward
                save_q_table(
                    q_table,
                    model_path.with_name(model_path.stem + "_best.npy"),
                    metadata={
                        "env_id": env_id,
                        "env_kwargs": env_kwargs,
                        "best_eval_mean_reward": best_eval_reward,
                        "episode": episode,
                    },
                )

        if checkpoint_freq > 0 and episode % checkpoint_freq == 0:
            checkpoint_path = model_path.with_name(f"{model_path.stem}_ep{episode}.npy")
            save_q_table(q_table, checkpoint_path)

    writer.close()

    save_q_table(
        q_table,
        model_path,
        metadata={
            "env_id": env_id,
            "env_kwargs": env_kwargs,
            "episodes": episodes,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "min_epsilon": min_epsilon,
            "max_epsilon": max_epsilon,
            "decay_rate": decay_rate,
            "max_steps": max_steps,
            "seed": seed,
            "success_threshold": success_threshold,
        },
    )
    print(f"Q-table saved to: {model_path}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Track live metrics with: uv run tensorboard --logdir {log_dir} --port 6006")
    return q_table


def rollout_to_gif(
    env: gym.Env,
    q_table: np.ndarray,
    gif_path: Path,
    max_steps: int,
    fps: int,
    seed: int,
) -> None:
    """Run one greedy episode and save frames as GIF."""
    state, _ = env.reset(seed=seed)

    frames: list[Image.Image] = []
    total_reward = 0.0

    first_frame = env.render()
    if first_frame is not None:
        frames.append(Image.fromarray(first_frame))

    for _ in range(1, max_steps + 1):
        action = greedy_policy(q_table, int(state))
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        if terminated or truncated:
            break

    if not frames:
        raise RuntimeError("No frames captured for GIF generation.")

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
    )
    print(f"GIF saved to: {gif_path} (reward: {total_reward:.2f}, frames: {len(frames)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic tabular Q-learning (discrete spaces) with TensorBoard + GIF export"
    )
    parser.add_argument("--train", action="store_true", help="Train a new Q-table")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate Q-table")
    parser.add_argument("--gif", action="store_true", help="Generate GIF rollout")

    parser.add_argument("--env-id", "--env", dest="env_id", type=str, default=DEFAULT_ENV_ID, help="Gymnasium env id")
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default='{"map_name":"8x8","is_slippery":true}',
        help="JSON dict of kwargs for gym.make, e.g. '{\"map_name\":\"8x8\",\"is_slippery\":true}'",
    )

    parser.add_argument("--train-episodes", type=int, default=200_000, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=1_000, help="Evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")

    parser.add_argument("--learning-rate", type=float, default=0.1, help="Q-learning learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    parser.add_argument("--max-epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="Final epsilon lower bound")
    parser.add_argument("--decay-rate", type=float, default=0.00005, help="Epsilon decay rate")

    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluate every N episodes")
    parser.add_argument("--checkpoint-freq", type=int, default=20_000, help="Checkpoint every N episodes (0 disables)")
    parser.add_argument("--success-threshold", type=float, default=0.0, help="Episode reward threshold counted as success")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to .npy Q-table")
    parser.add_argument("--gif-path", type=Path, default=None, help="Path to output GIF")
    parser.add_argument("--log-dir", type=Path, default=None, help="TensorBoard log directory")
    parser.add_argument("--fps", type=int, default=3, help="GIF FPS")

    args = parser.parse_args()
    if not (args.train or args.evaluate or args.gif):
        parser.error("Select at least one action: --train and/or --evaluate and/or --gif")
    return args


def main() -> None:
    args = parse_args()
    env_kwargs = parse_env_kwargs(args.env_kwargs)

    slug = env_slug(args.env_id)
    model_path = normalize_model_path(args.model_path or Path(f"models/qtable_{slug}.npy"))
    gif_path = args.gif_path or Path(f"videos/{slug}_test.gif")
    log_dir = args.log_dir or Path(f"logs/q_learning/{slug}")

    train_env = None
    eval_env = None
    gif_env = None
    try:
        if args.train:
            train_env = build_env(args.env_id, env_kwargs)
            eval_env = build_env(args.env_id, env_kwargs)
            q_table = train(
                env=train_env,
                eval_env=eval_env,
                env_id=args.env_id,
                env_kwargs=env_kwargs,
                episodes=args.train_episodes,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                min_epsilon=args.min_epsilon,
                max_epsilon=args.max_epsilon,
                decay_rate=args.decay_rate,
                max_steps=args.max_steps,
                model_path=model_path,
                log_dir=log_dir,
                eval_freq=args.eval_freq,
                checkpoint_freq=args.checkpoint_freq,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                success_threshold=args.success_threshold,
            )
        else:
            q_table = load_q_table(model_path)

        if args.evaluate:
            if eval_env is None:
                eval_env = build_env(args.env_id, env_kwargs)
            validate_q_table_shape(q_table, eval_env)

            mean_reward, std_reward, success_rate, mean_ep_len = evaluate_agent(
                env=eval_env,
                q_table=q_table,
                max_steps=args.max_steps,
                episodes=args.eval_episodes,
                seed=args.seed + 1000,
                success_threshold=args.success_threshold,
            )
            print(
                "Evaluation -> "
                f"mean_reward: {mean_reward:.4f}, std_reward: {std_reward:.4f}, "
                f"success_rate: {success_rate:.4f}, mean_ep_len: {mean_ep_len:.2f}"
            )

        if args.gif:
            gif_env = build_env(args.env_id, env_kwargs, render_mode="rgb_array")
            validate_q_table_shape(q_table, gif_env)
            rollout_to_gif(
                env=gif_env,
                q_table=q_table,
                gif_path=gif_path,
                max_steps=args.max_steps,
                fps=args.fps,
                seed=args.seed,
            )
    finally:
        for env in [train_env, eval_env, gif_env]:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# QUICK RUN / TEST COMMANDS (uv)
# -----------------------------------------------------------------------------
# FrozenLake (8x8 slippery):
#   uv run python3 q-learning.py --env-id FrozenLake-v1 \
#     --env-kwargs '{"map_name":"8x8","is_slippery":true}' --train
#
# Taxi:
#   uv run python3 q-learning.py --env-id Taxi-v3 --env-kwargs '{}' --train
#
# Evaluate + GIF (any supported discrete env):
#   uv run python3 q-learning.py --env-id Taxi-v3 --evaluate --gif
#
# TensorBoard:
#   uv add torch tensorboard
#   uv pip install 'setuptools<70'
#   uv run tensorboard --logdir logs/q_learning --port 6006
# Then open: http://localhost:6006


# -----------------------------------------------------------------------------
# TENSORBOARD METRICS CHEAT-SHEET (generic Q-learning)
# -----------------------------------------------------------------------------
# train/episode_reward
#   Reward in each training episode.
# train/reward_mean_100
#   Rolling mean reward across last 100 episodes.
# train/success
#   1 if episode reward >= --success-threshold else 0.
# train/success_rate_100
#   Rolling success ratio over last 100 episodes.
# train/epsilon
#   Exploration probability (epsilon-greedy).
# train/td_error_mean
#   Mean absolute TD error in the episode.
# train/q_value_mean, train/q_value_max
#   Learned Q-table magnitude trends.
# eval/mean_reward, eval/std_reward
#   Periodic evaluation quality metrics.
# eval/success_rate
#   Evaluation success ratio.
# eval/mean_episode_length
#   Average episode length during eval.
# -----------------------------------------------------------------------------
