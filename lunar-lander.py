from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


ENV_ID = "LunarLander-v3"
DEFAULT_MODEL_PATH = Path("models/ppo_lunar_lander")
DEFAULT_GIF_PATH = Path("videos/lunar_lander_test.gif")
DEFAULT_LOG_DIR = Path("logs/lunar_lander")


def build_model(env, device: str = "cpu", tensorboard_log: Path | None = None) -> PPO:
    """Create PPO model with sensible defaults for LunarLander."""
    return PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.005,
        verbose=1,
        device=device,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )


def train_model(
    timesteps: int,
    n_envs: int,
    seed: int,
    model_path: Path,
    device: str,
    log_dir: Path,
    eval_freq: int,
    checkpoint_freq: int,
    eval_episodes: int,
) -> PPO:
    """Train PPO, track metrics, and save model/checkpoints."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Monitor wrapper is required for robust reward/episode length logging.
    vec_env = make_vec_env(ENV_ID, n_envs=n_envs, seed=seed, wrapper_class=Monitor)
    model = build_model(vec_env, device=device, tensorboard_log=log_dir)

    # For vectorized envs, callback frequencies should be adjusted by n_envs.
    adjusted_eval_freq = max(eval_freq // max(1, n_envs), 1)
    adjusted_checkpoint_freq = max(checkpoint_freq // max(1, n_envs), 1)

    eval_env = Monitor(gym.make(ENV_ID))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_path.parent / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=adjusted_eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )

    callbacks: list = [eval_callback]

    if checkpoint_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=adjusted_checkpoint_freq,
            save_path=str(model_path.parent / "checkpoints"),
            name_prefix=model_path.name,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=CallbackList(callbacks),
        tb_log_name="ppo_lunar_lander",
    )

    model.save(str(model_path))
    eval_env.close()
    vec_env.close()

    print(f"Model saved to: {model_path}.zip")
    print(f"TensorBoard logs: {log_dir}")
    print(
        "Track live metrics with: "
        f"tensorboard --logdir {log_dir} --port 6006"
    )
    return model


def load_model(model_path: Path, device: str) -> PPO:
    """Load an existing PPO model."""
    zip_path = model_path.with_suffix(".zip")
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {zip_path}. Train first with --train."
        )

    model = PPO.load(str(model_path), device=device)
    print(f"Loaded model from: {zip_path}")
    return model


def evaluate(model: PPO, episodes: int, deterministic: bool = True) -> None:
    """Evaluate trained policy."""
    eval_env = Monitor(gym.make(ENV_ID))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=deterministic,
    )
    eval_env.close()
    print(f"Evaluation over {episodes} episodes -> mean: {mean_reward:.2f}, std: {std_reward:.2f}")


def rollout_to_gif(
    model: PPO,
    gif_path: Path,
    max_steps: int,
    fps: int,
    deterministic: bool,
    seed: int,
) -> None:
    """Run one episode and export it as a GIF."""
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    frames: list[Image.Image] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        action, _ = model.predict(obs, deterministic=deterministic)
        if isinstance(action, np.ndarray):
            action = int(action.item())

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    # Capture the final frame too.
    final_frame = env.render()
    if final_frame is not None:
        frames.append(Image.fromarray(final_frame))

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
    parser = argparse.ArgumentParser(description="LunarLander PPO trainer/evaluator with GIF export")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs for training")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="TensorBoard/eval log directory")
    parser.add_argument("--eval-freq", type=int, default=20_000, help="Evaluate every N timesteps during training")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint every N timesteps (0 disables)")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation episodes")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--gif", action="store_true", help="Generate a GIF test run")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max rollout steps for GIF episode")
    parser.add_argument("--fps", type=int, default=30, help="GIF FPS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions for eval/GIF")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Model path without .zip")
    parser.add_argument("--gif-path", type=Path, default=DEFAULT_GIF_PATH, help="Path to output GIF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = not args.stochastic

    if args.train:
        model = train_model(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            model_path=args.model_path,
            device=args.device,
            log_dir=args.log_dir,
            eval_freq=args.eval_freq,
            checkpoint_freq=args.checkpoint_freq,
            eval_episodes=args.eval_episodes,
        )
    else:
        model = load_model(args.model_path, device=args.device)

    if args.evaluate:
        evaluate(model, episodes=args.eval_episodes, deterministic=deterministic)

    if args.gif:
        rollout_to_gif(
            model=model,
            gif_path=args.gif_path,
            max_steps=args.max_steps,
            fps=args.fps,
            deterministic=deterministic,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# QUICK RUN / TEST COMMANDS (uv)
# -----------------------------------------------------------------------------
# Train (saves model + logs):
#   uv run python3 lunar-lander.py --train
#
# Train with custom setup:
#   uv run python3 lunar-lander.py --train --timesteps 1500000 --n-envs 8
#
# Evaluate saved model:
#   uv run python3 lunar-lander.py --evaluate --eval-episodes 10
#
# Generate GIF test rollout:
# (requires a trained model at models/ppo_lunar_lander.zip)
#   uv run python3 lunar-lander.py --gif
#
# Evaluate + GIF in one run:
#   uv run python3 lunar-lander.py --evaluate --gif

# Train and then immediately generate GIF:
#   uv run python3 lunar-lander.py --train --gif
#
# TensorBoard (install if missing):
#   uv add tensorboard
#   uv run tensorboard --logdir logs/lunar_lander --port 6006
# Then open: http://localhost:6006


# -----------------------------------------------------------------------------
# TENSORBOARD METRICS CHEAT-SHEET (PPO / Stable-Baselines3)
# -----------------------------------------------------------------------------
# rollout/ep_rew_mean
#   Mean episodic reward from training envs. Main learning signal.
#   Higher is better (for LunarLander, ~200+ is usually considered solved).
#
# rollout/ep_len_mean
#   Mean episode length. Can go up/down depending on behavior;
#   useful only with reward trend together.
#
# eval/mean_reward
#   Mean reward from periodic EvalCallback episodes (separate from train env).
#   Most reliable curve for true policy quality. Higher is better.
#
# eval/mean_ep_length
#   Avg evaluation episode length. Helpful context for eval/mean_reward.
#
# train/policy_gradient_loss
#   Policy update objective component. Magnitude/negative sign is normal;
#   focus on stability, not absolute value.
#
# train/value_loss
#   Critic (value function) fitting loss. Should be reasonably stable;
#   huge spikes can indicate instability.
#
# train/entropy_loss
#   Exploration proxy. Typically trends toward 0 as policy becomes confident.
#   If it collapses too fast, exploration may be too low.
#
# train/approx_kl
#   Approximate KL divergence between old/new policy after update.
#   Tracks update size; very large spikes mean overly aggressive updates.
#
# train/clip_fraction
#   Fraction of samples where PPO clipping activated.
#   Very high values can mean updates are too strong.
#
# train/explained_variance
#   How well value net explains returns. Closer to 1 is better.
#   Around 0 or negative means poor value predictions.
#
# train/loss
#   Combined optimization loss (policy + value + entropy terms).
#   Useful as sanity signal; less interpretable than reward/eval curves.
#
# train/learning_rate
#   Current LR used by optimizer.
#
# time/fps, time/iterations, time/total_timesteps
#   Throughput/progress metrics (speed and training progress), not quality.
# -----------------------------------------------------------------------------