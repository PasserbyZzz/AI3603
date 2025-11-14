# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    # Whether to record training episodes
    parser.add_argument("--capture-video", type=lambda x: bool(int(x)), default=1,
        help="set to 1 to record final video (saved in videos/)" )
    # Model checkpoint save frequency (in env steps)
    parser.add_argument("--save-model-frequency", type=int, default=50000,
        help="save model checkpoint every N environment steps")
    # Directory to store model checkpoints
    parser.add_argument("--model-dir", type=str, default="models",
        help="directory to store model checkpoints")
    # Number of greedy evaluation episodes to record at the end
    parser.add_argument("--eval-episodes", type=int, default=5,
        help="number of greedy evaluation episodes to record at the end")
    
    # Algorithm specific arguments
    # Experiment name
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    # Total training timesteps
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    # Learning rate
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    # Replay buffer size
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    # Discount factor gamma
    parser.add_argument("--gamma", type=float, default=0.6,
        help="the discount factor gamma")
    # Target network update frequency (in env steps)
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    # Batch size sampled from the replay buffer each update
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    # Starting epsilon value (exploration rate)
    parser.add_argument("--start-e", type=float, default=0.3,
        help="the starting epsilon for exploration")
    # Final epsilon value after decay
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    # Fraction of total timesteps used for linear epsilon decay
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    # Warm-up timesteps before learning starts (fill buffer)
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    # Training frequency in environment steps
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
   
    args = parser.parse_args()
    # Environment id
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed, capture_video=False, run_name=None):
    """Construct the gym environment; optionally wrap with RecordVideo.

    Args:
        env_id: gym environment id.
        seed: random seed for reproducibility.
        capture_video: whether to add a video recording wrapper.
        run_name: folder suffix (usually includes timestamp) used for organizing videos.
    """
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """A feed-forward Q-network (MLP) mapping state -> action values.

    - Input: state vector of shape (obs_dim,), obs_dim = env.observation_space.shape[0].
    - Output: Q(s, ·) of shape (n_actions,), n_actions = env.action_space.n.
    - Architecture: 120 -> 84 hidden units with ReLU activations.
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linearly decay epsilon from start_e to end_e within `duration` steps.

    Args:
        start_e: initial epsilon (exploration rate).
        end_e: minimum epsilon after decay.
        duration: number of timesteps to complete the decay.
        t: current global timestep.

    Returns:
        Epsilon value at step t, clipped to be no less than end_e.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def save_model(q_network: QNetwork, global_step: int, args, run_name: str, tag: str = "step"):
    """Persist model parameters to disk in `models/` directory.

    File name pattern: <run_name>_<tag>_<global_step>.pt storing state_dict
    plus metadata (hyperparameters and global step).
    """
    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, f"{run_name}_{tag}_{global_step}.pt")
    torch.save({
        "global_step": global_step,
        "model_state_dict": q_network.state_dict(),
        "args": vars(args),
    }, path)
    return path

def evaluate_and_record(env_id: str, q_network: QNetwork, device, run_name: str, episodes: int, video: bool = True):
    """Run greedy (epsilon=0) evaluation episodes; optionally record video.

    Returns a list of episodic returns for reporting.
    """
    eval_dir = f"videos/{run_name}" if video else None
    env = gym.make(env_id)
    if video:
        os.makedirs(eval_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, eval_dir)
    returns = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                q_vals = q_network(torch.Tensor(obs).to(device))
                action = torch.argmax(q_vals, dim=0).item()
            obs, reward, done, info = env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
    env.close()
    return returns

if __name__ == "__main__":
    
    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    """we utilize tensorboard to log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    """Set RNG seeds and choose compute device for reproducibility.

    Fix random seeds of Python/NumPy/PyTorch, enable deterministic cuDNN, and
    select CUDA when available (else fall back to CPU).
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Create the Gym environment and wrap it to record episode statistics.

    Apply the same seed to the environment, action space, and observation space
    to make results reproducible across runs.
    """
    envs = make_env(args.env_id, args.seed, capture_video=bool(args.capture_video), run_name=run_name)

    """Instantiate online and target Q-networks and the optimizer.

    The target network starts as a copy of the online network and will be
    periodically synchronized to stabilize bootstrap targets.
    """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """Initialize the replay buffer that stores transitions for off-policy training."""
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """Reset the environment and start the main interaction/training loop."""
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        
        """Compute epsilon according to the linear schedule for exploration."""
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        """Epsilon-greedy policy: random action with prob epsilon, otherwise argmax Q."""
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        
        """Interact with the environment; log episodic return/length when done."""
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training
        
        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
        
        """Store the transition (s, a, r, s', done, info) into the replay buffer."""
        rb.add(obs, next_obs, actions, rewards, dones, infos)
        
        """Advance to the next observation; reset the env if the episode ended."""
        obs = next_obs if not dones else envs.reset()
        
        """Begin training."""
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            
            """It's training time: sample a batch from the buffer and update networks."""
            data = rb.sample(args.batch_size)
            
            """Compute the TD target with the target network and the MSE loss."""
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """Log loss and mean Q-value every 100 steps for monitoring."""
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            
            """Backpropagate and update the online Q-network parameters."""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            """Periodically sync the target network with the online network."""
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            # Periodic model checkpoint
            if global_step % args.save_model_frequency == 0:
                ckpt_path = save_model(q_network, global_step, args, run_name)
                print(f"Saved model checkpoint to: {ckpt_path}")
    
    # save final model
    final_path = save_model(q_network, global_step=args.total_timesteps, args=args, run_name=run_name, tag="final")
    print(f"Final model saved to: {final_path}")

    # record final evaluation video
    if args.capture_video:
        eval_returns = evaluate_and_record(args.env_id, q_network, device, run_name, episodes=args.eval_episodes, video=True)
        print(f"Final evaluation returns: {eval_returns}")

    """close the env and tensorboard logger"""
    envs.close()
    writer.close()