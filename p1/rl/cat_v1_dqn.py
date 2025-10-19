#max reward 500 is achieved
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 输入层
        self.fc2 = nn.Linear(128, 128)        # 隐藏层
        self.fc3 = nn.Linear(128, output_dim) # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数
        x = F.relu(self.fc2(x))
        return self.fc3(x)       # 输出每个动作的Q值，不经过激活函数




import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch.optim as optim



# 超参数
BATCH_SIZE = 32
LR = 0.001
GAMMA = 0.99  # 折扣因子:cite[5]
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

# 初始化环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 初始化网络
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())  # 同步目标网络参数
target_net.eval()  # 目标网络设置为评估模式

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=10000)  # 经验回放缓冲区

epsilon = EPSILON_START

# 训练循环
for episode in range(1000):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):  # CartPole-v1最多500步
        # ε-贪婪策略选择动作:cite[5]
        if random.random() < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()  # 利用

        # 执行动作
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        # 存储经验
        memory.append((state, action, reward, next_state, done))

        state = next_state

        # 经验回放与训练
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # 转换为Tensor
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            # 计算当前Q值
            current_q_values = policy_net(states).gather(1, actions)
            # 计算目标Q值
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + (GAMMA * next_q_values * ~dones)
            target_q_values = target_q_values.unsqueeze(1)

            # 计算损失并更新网络
            loss = F.mse_loss(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done or truncated:
            break

    # 更新探索率
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # 更新目标网络
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()