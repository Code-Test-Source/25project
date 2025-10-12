import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 创建保存模型的目录
if not os.path.exists('sac_model'):
    os.makedirs('sac_model')


# 定义Actor网络（策略网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()  # 使用重参数化技巧
        action = torch.tanh(x_t)

        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        # 修正由于tanh变换导致的概率变化
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


# 定义Critic网络（价值网络）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1网络
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)

        # Q2网络
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Q1值
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        # Q2值
        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)

        return q1, q2


# 定义SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=1000000, batch_size=256, auto_entropy_tuning=True):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning

        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # 复制参数到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 自动调整熵权重
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=buffer_size)

        # 训练步数计数器
        self.total_steps = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate:
            with torch.no_grad():
                mean, log_std = self.actor(state)
                action = torch.tanh(mean)
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0

        # 从经验回放中采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device).unsqueeze(1)

        # 更新Critic网络
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * next_q * (~dones)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        actions_pi, log_probs_pi = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, actions_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_probs_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 自动调整熵权重
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_steps += 1

        return critic_loss.item(), actor_loss.item(), alpha_loss.item() if self.auto_entropy_tuning else 0

    def save_models(self, episode):
        torch.save(self.actor.state_dict(), f'sac_model/actor_{episode}.pth')
        torch.save(self.critic.state_dict(), f'sac_model/critic_{episode}.pth')
        torch.save(self.critic_target.state_dict(), f'sac_model/critic_target_{episode}.pth')

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(f'sac_model/actor_{episode}.pth'))
        self.critic.load_state_dict(torch.load(f'sac_model/critic_{episode}.pth'))
        self.critic_target.load_state_dict(torch.load(f'sac_model/critic_target_{episode}.pth'))


# 训练函数
def train_sac(env_name="BipedalWalker-v3", num_episodes=2000, max_steps=1600):
    # 创建环境
    env = gym.make(env_name)

    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")

    # 初始化Agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256
    )

    # 记录训练过程
    episode_rewards = []
    episode_lengths = []
    critic_losses = []
    actor_losses = []
    alpha_losses = []

    best_reward = -float('inf')

    # 训练循环
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        total_critic_loss = 0
        total_actor_loss = 0
        total_alpha_loss = 0
        steps = 0

        for step in range(max_steps):
            # 选择并执行动作
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)

            # 训练
            critic_loss, actor_loss, alpha_loss = agent.train()
            total_critic_loss += critic_loss if critic_loss else 0
            total_actor_loss += actor_loss if actor_loss else 0
            total_alpha_loss += alpha_loss if alpha_loss else 0

            state = next_state
            total_reward += reward
            steps += 1

            if done or truncated:
                break

        # 记录结果
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        critic_losses.append(total_critic_loss / steps if steps > 0 else 0)
        actor_losses.append(total_actor_loss / steps if steps > 0 else 0)
        alpha_losses.append(total_alpha_loss / steps if steps > 0 else 0)

        # 保存最佳模型
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_models('best')

        # 打印进度
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Average Reward (last 50): {avg_reward:.2f}, Steps: {steps}")

            # 每100个回合保存一次模型
            if episode % 100 == 0 and episode > 0:
                agent.save_models(episode)

    # 绘制训练曲线
    plot_training_results(episode_rewards, critic_losses, actor_losses, alpha_losses, episode_lengths)

    env.close()
    return agent, episode_rewards


# 测试训练好的Agent
def test_agent(agent, env_name="BipedalWalker-v3", num_episodes=5, render=True):
    env = gym.make(env_name, render_mode="human" if render else None)
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()
    print(f"Average Test Reward: {np.mean(total_rewards):.2f}")
    return total_rewards


# 绘制训练结果
def plot_training_results(rewards, critic_losses, actor_losses, alpha_losses, lengths):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 绘制奖励
    ax1.plot(rewards)
    ax1.set_title('Training Rewards')
    ax1.set_ylabel('Reward')
    ax1.grid(True)

    # 绘制平滑后的奖励（使用移动平均）
    window_size = 50
    if len(rewards) >= window_size:
        smooth_rewards = [np.mean(rewards[i:i + window_size]) for i in range(len(rewards) - window_size)]
        ax1.plot(range(window_size, len(rewards)), smooth_rewards, 'r-', linewidth=2, label='Smoothed')
        ax1.legend()

    # 绘制损失
    ax2.plot(critic_losses, label='Critic Loss')
    ax2.plot(actor_losses, label='Actor Loss')
    ax2.set_title('Training Losses')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Episode')
    ax2.legend()
    ax2.grid(True)

    # 绘制alpha损失
    ax3.plot(alpha_losses)
    ax3.set_title('Alpha Loss')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Episode')
    ax3.grid(True)

    # 绘制回合长度
    ax4.plot(lengths)
    ax4.set_title('Episode Lengths')
    ax4.set_ylabel('Steps')
    ax4.set_xlabel('Episode')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('sac_training_results.png')
    plt.show()


# 主函数
if __name__ == "__main__":
    print("Starting SAC training on BipedalWalker-v3...")

    # 训练Agent
    agent, rewards = train_sac(
        env_name="BipedalWalker-v3",
        num_episodes=2000,
        max_steps=1600
    )

    # 加载最佳模型并测试
    print("\nLoading best model and testing...")
    agent.load_models('best')
    test_agent(agent, "BipedalWalker-v3", num_episodes=5, render=True)