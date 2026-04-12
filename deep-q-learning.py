import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py
import sys
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import os
import time
from PIL import Image

gym.register_envs(ale_py)

# hyperparams

ENV_NAME = "ALE/SpaceInvaders-v5"
SEED = 0
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
REPLAY_SIZE = 100_000
MIN_REPLAY = 50_000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_FRAMES = 1_000_000
TARGET_UPDATE_FREQ = 10_000
TOTAL_FRAMES = 2_000_000
EVAL_FREQ = 50_000
EVAL_EPISODES = 5
SAVE_FREQ = 500_000

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
DEVICE = get_device()
print(f"Using device: {DEVICE}")

class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env, frame_skip=4, stack_size=4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(stack_size, 84, 84), dtype=np.float32
        )

    def _preprocess(self, obs):
        gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
        h, w = gray.shape
        row_idx = (np.arange(84) * h / 84).astype(int)
        col_idx = (np.arange(84) * w / 84).astype(int)
        resized = gray[np.ix_(row_idx, col_idx)]
        return resized / 255.0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._preprocess(obs)
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return self._stack(), info
    
    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(self.frame_skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        self.frames.append(self._preprocess(obs))
        return self._stack(), total_reward, done, truncated, info
    
    def _stack(self):
        return np.array(self.frames, dtype=np.float32)
    
def make_env(render_mode=None):
    env = gym.make(ENV_NAME, render_mode=render_mode)
    return AtariPreprocessing(env)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self):
        return len(self.buffer)
    
class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
def get_epsilon(frame_idx):
    progress = min(frame_idx / EPSILON_DECAY_FRAMES, 1.0)
    return EPSILON_START + (EPSILON_END - EPSILON_START) * progress

def train(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log_dir = Path(args.output_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models/best_model", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    print(f'tensorboard logs: {log_dir}')

    env = make_env()
    n_actions = env.action_space.n
    print(f'total no of actions: {n_actions}')

    writer.add_text("hyperparams", f"""
| Param | Value |
|---|---|
| env | {ENV_NAME} |
| gamma | {GAMMA} |
| lr | {LR} |
| batch_size | {BATCH_SIZE} |
| replay_size | {REPLAY_SIZE} |
| epsilon_decay_frames | {EPSILON_DECAY_FRAMES} |
| target_update_freq | {TARGET_UPDATE_FREQ} |
| total_frames | {TOTAL_FRAMES} |
| device | {DEVICE} |
""")

    q_net = QNetwork(n_actions).to(DEVICE)
    target_net = QNetwork(n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)

    episode_rewards = []
    episode_lengths = []
    current_reward = 0.0
    current_length = 0
    best_eval_reward = -float("inf")
    episode_count = 0
    train_steps = 0
    t_start = time.time()

    state, _ = env.reset(seed=SEED)
    frame_idx = 0

    while frame_idx < TOTAL_FRAMES:
        eps = get_epsilon(frame_idx)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action = q_net(s_t).argmax(dim=1).item()
        
        next_state, reward, done, truncated, _ = env.step(action)
        replay.push(state, action, reward, next_state, done or truncated)
        state = next_state
        current_reward += reward
        current_length += 1
        frame_idx += 1

        if done or truncated:
            episode_count += 1
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)

            writer.add_scalar("episode/reward", current_reward, frame_idx)
            writer.add_scalar("episode/length", current_length, frame_idx)
            writer.add_scalar("episode/reward_avg100",
                              np.mean(episode_rewards[-100:]), frame_idx)
            if len(episode_rewards) >= 10:
                writer.add_scalar("episode/reward_avg10",
                                  np.mean(episode_rewards[-10:]), frame_idx)
                
            state, _ = env.reset()
            current_reward = 0.0
            current_length = 0

        if len(replay) < MIN_REPLAY:
            if frame_idx % 10_000 == 0:
                writer.add_scalar("buffer/size", len(replay), frame_idx)
            continue

        states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

        states_t = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

        q_values = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = target_net(next_states_t).max(dim=1).values
            target = rewards_t + GAMMA * next_q * (1.0 - dones_t)

        loss = nn.functional.huber_loss(q_values, target)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()
        train_steps += 1

        if train_steps % 1000 == 0:
            writer.add_scalar("train/loss", loss.item(), frame_idx)
            writer.add_scalar("train/q_mean", q_values.mean().item(), frame_idx)
            writer.add_scalar("train/q_max", q_values.max().item(), frame_idx)
            writer.add_scalar("train/grad_norm", grad_norm.item(), frame_idx)
            writer.add_scalar("train/epsilon", eps, frame_idx)
            writer.add_scalar("buffer/size", len(replay), frame_idx)

            elapsed = time.time() - t_start
            fps = frame_idx / elapsed
            writer.add_scalar("perf/fps", fps, frame_idx)

        if frame_idx % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f'sync target at frame {frame_idx}')
        
        if frame_idx % EVAL_FREQ == 0:
            avg = evaluate(q_net, n_actions)
            last_10 = np.mean(episode_rewards[-10:]) if episode_rewards else 0

            elapsed = time.time() - t_start
            fps = frame_idx / elapsed

            writer.add_scalar("eval/reward_avg", avg, frame_idx)

            print(
                f"frame {frame_idx:>8d} | "
                f"eps {eps:.3f} | "
                f"train(last10) {last_10:>7.1f} | "
                f"eval(avg{EVAL_EPISODES}) {avg:>7.1f} | "
                f"buffer {len(replay)}"
            )
            if avg > best_eval_reward:
                best_eval_reward = avg
                torch.save(q_net.state_dict(), "models/best_model/best_model_qnet.pt")
                writer.add_scalar("eval/best_reward", best_eval_reward, frame_idx)

        if frame_idx % SAVE_FREQ == 0:
            torch.save(q_net.state_dict(), f"models/checkpoints/checkpoint_{frame_idx}.pt")

    env.close()
    torch.save(q_net.state_dict(), "models/final_model.pt")
    writer.close()
    print(f'training done')

def evaluate(q_net, n_actions, episodes=EVAL_EPISODES):
    env = make_env()
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            if random.random() < 0.05:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    action = q_net(s_t).argmax(dim=1).item()
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            if truncated:
                break
        total_rewards.append(ep_reward)
    env.close()
    return np.mean(total_rewards)

def watch(args):
    env = make_env(render_mode="human")
    n_actions = env.action_space.n
    q_net = QNetwork(n_actions).to(DEVICE)
    model_path = args.model or Path("models/best_model/best_model_qnet.pt")
    q_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    q_net.eval()

    state, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = q_net(s_t).argmax(dim=1).item()
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if truncated:
            break
    print(f'score: {total_reward}')
    env.close()

def save_gif(args):
    env = make_env(render_mode="rgb_array")
    n_actions = env.action_space.n
    q_net = QNetwork(n_actions).to(DEVICE)
    model_path = Path(args.model or "models/best_model/best_model_qnet.pt")
    q_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    q_net.eval()

    frames = []
    state, _ = env.reset(seed=SEED)
    frame = env.render()
    if frame is not None:
        frames.append(Image.fromarray(frame))

    total_reward = 0.0
    done = False
    truncated = False
    steps = 0

    while not (done or truncated) and steps < args.max_steps:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = q_net(s_t).argmax(dim=1).item()
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

    env.close()

    if not frames:
        raise RuntimeError("No frames captured for GIF generation.")

    gif_path = Path(args.gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / max(1, args.fps)),
        loop=0,
    )
    print(f"GIF saved to: {gif_path} (reward: {total_reward:.2f}, frames: {len(frames)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train DQN model")
    parser.add_argument("--watch", action="store_true", help="Watch one gameplay episode")
    parser.add_argument("--gif", action="store_true", help="Save one gameplay episode as GIF")

    parser.add_argument("--output-dir", default="logs/dqn")
    parser.add_argument("--model", default="models/best_model/best_model_qnet.pt")
    parser.add_argument("--gif-path", default="videos/deep_q_space_invaders.gif")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=5000)

    args = parser.parse_args()
    if not (args.train or args.watch or args.gif):
        parser.error("Select at least one action: --train and/or --watch and/or --gif")

    if args.train:
        train(args)
    if args.watch:
        watch(args)
    if args.gif:
        save_gif(args)


if __name__ == "__main__":
    main()
