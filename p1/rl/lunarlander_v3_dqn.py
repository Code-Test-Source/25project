import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# 设置随机种子以便复现结果
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 定义DQN神经网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# 定义DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # 设备选择 (GPU如果可用，否则CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 初始化策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络设置为评估模式

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)

        # 训练步数计数器
        self.steps_done = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def update_epsilon(self):
        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0  # 经验不足，不进行训练

        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            target_q_values = target_q_values.unsqueeze(1)

        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        # 更新目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 训练函数
def train_dqn(env_name="LunarLander-v3", num_episodes=1500, max_steps=1000,
              save_model=True, render_freq=100):
    # 创建环境
    env = gym.make(env_name)

    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # 初始化Agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update=10
    )

    # 记录训练过程
    episode_rewards = []
    episode_losses = []
    episode_lengths = []

    # 训练循环
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0

        for step in range(max_steps):
            # 选择并执行动作
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)

            # 训练
            loss = agent.train_step()
            total_loss += loss if loss else 0

            state = next_state
            total_reward += reward
            steps += 1

            # 更新目标网络
            if agent.steps_done % agent.target_update == 0:
                agent.update_target_network()

            agent.steps_done += 1

            if done or truncated:
                break

        # 更新探索率
        agent.update_epsilon()

        # 记录结果
        episode_rewards.append(total_reward)
        episode_losses.append(total_loss / steps if steps > 0 else 0)
        episode_lengths.append(steps)

        # 打印进度
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Average Reward (last 50): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {steps}")

        # 定期渲染环境以查看进展
        if episode % render_freq == 0 and episode > 0:
            test_agent(agent, env_name, num_episodes=1, render=True)

    # 保存模型
    if save_model:
        torch.save(agent.policy_net.state_dict(), "dqn_lunar_lander.pth")
        print("Model saved as dqn_lunar_lander.pth")

    # 绘制训练曲线
    plot_training_results(episode_rewards, episode_losses, episode_lengths)

    env.close()
    return agent, episode_rewards


# 测试训练好的Agent
def test_agent(agent, env_name, num_episodes=10, render=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()
    print(f"Average Test Reward: {np.mean(total_rewards):.2f}")
    return total_rewards


# 绘制训练结果
def plot_training_results(rewards, losses, lengths):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # 绘制奖励
    ax1.plot(rewards)
    ax1.set_title('Training Rewards')
    ax1.set_ylabel('Reward')
    ax1.grid(True)

    # 绘制平滑后的奖励（使用移动平均）
    window_size = 50
    smooth_rewards = [np.mean(rewards[i:i + window_size]) for i in range(len(rewards) - window_size)]
    ax1.plot(range(window_size, len(rewards)), smooth_rewards, 'r-', linewidth=2, label='Smoothed')
    ax1.legend()

    # 绘制损失
    ax2.plot(losses)
    ax2.set_title('Training Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Episode')
    ax2.grid(True)

    # 绘制回合长度
    ax3.plot(lengths)
    ax3.set_title('Episode Lengths')
    ax3.set_ylabel('Steps')
    ax3.set_xlabel('Episode')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


# 主函数
if __name__ == "__main__":
    # 训练Agent
    print("Starting DQN training on LunarLander-v3...")
    agent, rewards = train_dqn(
        env_name="LunarLander-v3",
        num_episodes=1500,
        max_steps=1000,
        save_model=True,
        render_freq=200
    )

    # 测试训练好的Agent
    print("\nTesting trained agent...")
    test_agent(agent, "LunarLander-v3", num_episodes=5, render=True)