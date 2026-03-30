from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter


ENV_ID = "FrozenLake-v1"
DEFAULT_MAP_NAME = "8x8"
DEFAULT_MODEL_PATH = Path("models/qtable_frozenlake.npy")
DEFAULT_GIF_PATH = Path("videos/frozenlake_test.gif")
DEFAULT_LOG_DIR = Path("logs/frozenlake_qlearning")


def build_env(map_name: str, is_slippery: bool, render_mode: str | None = None):
    """Create FrozenLake env."""
    env = gym.make(
        ENV_ID,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode,
    )
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise TypeError("This script expects a discrete observation space.")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("This script expects a discrete action space.")
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
    q_table: np.ndarray,
    map_name: str,
    is_slippery: bool,
    max_steps: int,
    episodes: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Evaluate greedy policy and return mean/std reward, success rate, mean ep length."""
    env = build_env(map_name=map_name, is_slippery=is_slippery, render_mode=None)
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
                successes.append(1.0 if reward > 0 else 0.0)
                break
        else:
            lengths.append(max_steps)
            successes.append(0.0)

        rewards.append(total_reward)

    env.close()
    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        float(np.mean(successes)),
        float(np.mean(lengths)),
    )


def save_q_table(q_table: np.ndarray, model_path: Path, metadata: dict | None = None) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(model_path, q_table)
    if metadata:
        meta_path = model_path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2))


def load_q_table(model_path: Path) -> np.ndarray:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Q-table not found at: {model_path}. Train first with --train."
        )
    q_table = np.load(model_path)
    if q_table.ndim != 2:
        raise ValueError(f"Invalid Q-table shape: {q_table.shape}")
    return q_table.astype(np.float32)


def train(
    episodes: int,
    learning_rate: float,
    gamma: float,
    min_epsilon: float,
    max_epsilon: float,
    decay_rate: float,
    max_steps: int,
    map_name: str,
    is_slippery: bool,
    model_path: Path,
    log_dir: Path,
    eval_freq: int,
    checkpoint_freq: int,
    eval_episodes: int,
    seed: int,
) -> np.ndarray:
    """Train Q-learning agent with TensorBoard tracking."""
    env = build_env(map_name=map_name, is_slippery=is_slippery, render_mode=None)
    state_space = env.observation_space.n
    action_space = env.action_space.n
    q_table = initialize_q_table(state_space, action_space)

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    rng = np.random.default_rng(seed)

    reward_window: list[float] = []
    success_window: list[float] = []
    best_eval_reward = -np.inf

    pbar = tqdm(range(1, episodes + 1), desc="Training Q-learning")
    for episode in pbar:
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * (episode - 1))

        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        td_errors: list[float] = []
        success = 0.0

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
                success = 1.0 if reward > 0 else 0.0
                break

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
                q_table=q_table,
                map_name=map_name,
                is_slippery=is_slippery,
                max_steps=max_steps,
                episodes=eval_episodes,
                seed=seed + 10_000,
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
                        "env_id": ENV_ID,
                        "map_name": map_name,
                        "is_slippery": is_slippery,
                        "best_eval_mean_reward": best_eval_reward,
                        "episode": episode,
                    },
                )

        if checkpoint_freq > 0 and episode % checkpoint_freq == 0:
            checkpoint_path = model_path.with_name(
                f"{model_path.stem}_ep{episode}.npy"
            )
            save_q_table(q_table, checkpoint_path)

    writer.close()
    env.close()

    save_q_table(
        q_table,
        model_path,
        metadata={
            "env_id": ENV_ID,
            "map_name": map_name,
            "is_slippery": is_slippery,
            "episodes": episodes,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "min_epsilon": min_epsilon,
            "max_epsilon": max_epsilon,
            "decay_rate": decay_rate,
            "max_steps": max_steps,
            "seed": seed,
        },
    )
    print(f"Q-table saved to: {model_path}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Track live metrics with: uv run tensorboard --logdir {log_dir} --port 6006")
    return q_table


def rollout_to_gif(
    q_table: np.ndarray,
    map_name: str,
    is_slippery: bool,
    gif_path: Path,
    max_steps: int,
    fps: int,
    seed: int,
) -> None:
    """Run one greedy episode and save frames as GIF."""
    env = build_env(map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)

    frames: list[Image.Image] = []
    total_reward = 0.0

    first_frame = env.render()
    if first_frame is not None:
        frames.append(Image.fromarray(first_frame))

    for step in range(1, max_steps + 1):
        action = greedy_policy(q_table, int(state))
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        if terminated or truncated:
            break

    env.close()

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
        description="FrozenLake Q-learning trainer/evaluator with TensorBoard + GIF export"
    )
    parser.add_argument("--train", action="store_true", help="Train a new Q-table")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate Q-table")
    parser.add_argument("--gif", action="store_true", help="Generate a GIF rollout")

    parser.add_argument("--train-episodes", type=int, default=200_000, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=1_000, help="Evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")

    parser.add_argument("--learning-rate", type=float, default=0.1, help="Q-learning learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    parser.add_argument("--max-epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="Final epsilon lower bound")
    parser.add_argument("--decay-rate", type=float, default=0.00005, help="Epsilon decay rate")

    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluate every N episodes during training")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20_000,
        help="Save Q-table checkpoint every N episodes (0 disables)",
    )

    parser.add_argument("--map-name", type=str, default=DEFAULT_MAP_NAME, help="FrozenLake map name (4x4 or 8x8)")
    parser.add_argument("--no-slippery", action="store_true", help="Disable slippery dynamics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to .npy Q-table")
    parser.add_argument("--gif-path", type=Path, default=DEFAULT_GIF_PATH, help="Path to output GIF")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="TensorBoard log directory")
    parser.add_argument("--fps", type=int, default=3, help="GIF FPS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_slippery = not args.no_slippery

    if args.train:
        q_table = train(
            episodes=args.train_episodes,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            min_epsilon=args.min_epsilon,
            max_epsilon=args.max_epsilon,
            decay_rate=args.decay_rate,
            max_steps=args.max_steps,
            map_name=args.map_name,
            is_slippery=is_slippery,
            model_path=args.model_path,
            log_dir=args.log_dir,
            eval_freq=args.eval_freq,
            checkpoint_freq=args.checkpoint_freq,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
    else:
        q_table = load_q_table(args.model_path)

    if args.evaluate:
        mean_reward, std_reward, success_rate, mean_ep_len = evaluate_agent(
            q_table=q_table,
            map_name=args.map_name,
            is_slippery=is_slippery,
            max_steps=args.max_steps,
            episodes=args.eval_episodes,
            seed=args.seed + 1000,
        )
        print(
            "Evaluation -> "
            f"mean_reward: {mean_reward:.4f}, std_reward: {std_reward:.4f}, "
            f"success_rate: {success_rate:.4f}, mean_ep_len: {mean_ep_len:.2f}"
        )

    if args.gif:
        rollout_to_gif(
            q_table=q_table,
            map_name=args.map_name,
            is_slippery=is_slippery,
            gif_path=args.gif_path,
            max_steps=args.max_steps,
            fps=args.fps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# QUICK RUN / TEST COMMANDS (uv)
# -----------------------------------------------------------------------------
# Train (saves Q-table + logs):
#   uv run python3 q-learning.py --train
#
# Train with custom setup:
#   uv run python3 q-learning.py --train --train-episodes 300000 --map-name 8x8
#
# Evaluate saved Q-table:
#   uv run python3 q-learning.py --evaluate --eval-episodes 1000
#
# Generate GIF test rollout:
# (requires a trained model at models/qtable_frozenlake.npy)
#   uv run python3 q-learning.py --gif
#
# Train and then immediately generate GIF:
#   uv run python3 q-learning.py --train --gif
#
# Evaluate + GIF in one run:
#   uv run python3 q-learning.py --evaluate --gif
#
# TensorBoard (install if missing):
#   uv add torch tensorboard
#   uv pip install 'setuptools<70'
#   uv run tensorboard --logdir logs/frozenlake_qlearning --port 6006
# Then open: http://localhost:6006


# -----------------------------------------------------------------------------
# TENSORBOARD METRICS CHEAT-SHEET (Q-learning)
# -----------------------------------------------------------------------------
# train/episode_reward
#   Reward achieved in each training episode (0 or 1 on FrozenLake).
#
# train/reward_mean_100
#   Rolling mean reward over last 100 episodes.
#   Main trend for gradual learning quality.
#
# train/success
#   Episode success flag (1 if goal reached, else 0).
#
# train/success_rate_100
#   Rolling success rate over last 100 episodes.
#   Very useful for sparse-reward tasks like FrozenLake.
#
# train/epsilon
#   Exploration probability. Starts high, decays over time.
#
# train/td_error_mean
#   Mean absolute TD error in episode.
#   Should generally reduce as values become consistent.
#
# train/q_value_mean, train/q_value_max
#   Magnitude/trend of learned Q-values.
#
# train/episode_length
#   Steps taken per episode. Interpret with reward/success trends.
#
# eval/mean_reward, eval/std_reward
#   Performance from periodic greedy evaluation runs.
#
# eval/success_rate
#   Evaluation success ratio. Best quality indicator.
#
# eval/mean_episode_length
#   Average evaluation episode length.
# -----------------------------------------------------------------------------
