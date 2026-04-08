from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def build_model(env, tensorboard_log: Path) -> PPO:
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
        device="cpu",
        tensorboard_log=str(tensorboard_log)
    )

def train_model(
        timesteps: int,
        n_envs: int,
        model_path: Path,
        log_dir: Path,
        eval_freq: int,
        eval_episodes: int,
) -> PPO:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env("LunarLander-v3", n_envs=n_envs, seed=0, wrapper_class=Monitor)
    model = build_model(vec_env, tensorboard_log=log_dir)

    adjusted_eval_freq = max(eval_freq // max(1, n_envs), 1)

    eval_env = Monitor(gym.make("LunarLander-v3"))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_path.parent / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=adjusted_eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )

    callbacks: list = [eval_callback]

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

def load_model(model_path: Path) -> PPO:
    """Load an existing PPO model."""
    zip_path = model_path.with_suffix(".zip")
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {zip_path}. Train first with --train."
        )

    model = PPO.load(str(model_path), device="cpu")
    print(f"Loaded model from: {zip_path}")
    return model

def rollout_to_gif(
        model: PPO,
        gif_path: Path,
        max_steps: int,
        fps: int,
):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)

    frames: list[Image.Image] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        action, _ = model.predict(obs, deterministic=True)

        if isinstance(action, np.ndarray):
            action = int(action.item())

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    final_frame = env.render()
    if final_frame is not None:
        frames.append(Image.fromarray(final_frame))

    env.close()

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
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="train a new model")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--gif", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.train:
        model = train_model(
            timesteps=args.timesteps,
            n_envs=16,
            model_path=Path("models/ppo_lunar_lander"),
            log_dir=Path("logs/lunar_lander"),
            eval_freq=20_000,
            eval_episodes=10
        )
    else:
        model = load_model(Path("models/ppo_lunar_lander"))

    if args.gif:
        rollout_to_gif(
            model=model,
            gif_path=Path("videos/lunar_lander_test.gif"),
            max_steps=1000,
            fps=30
        )

if __name__ == "__main__":
    main()
