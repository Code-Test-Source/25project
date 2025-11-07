

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import os
import ale_py

gym.register_envs(ale_py)
# 设置OpenMP环境变量避免冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ImprovedCNN(BaseFeaturesExtractor):
    """
    改进的CNN特征提取器，更深的网络结构
    """

    def __init__(self, observation_space, features_dim=512):
        super(ImprovedCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算全连接层输入维度
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

        # 设置特征维度
        self._features_dim = features_dim

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def create_improved_atari_env(env_name='ALE/Pong-v5', frame_stack=4):
    """
    创建改进的Atari环境
    """

    def _init():
        try:
            # 创建环境
            env = gym.make(env_name,
                           render_mode='rgb_array',
                           frameskip=4,
                           full_action_space=False)

            # 应用标准预处理
            env = gym.wrappers.AtariPreprocessing(
                env,
                noop_max=30,  # 随机no-op操作
                frame_skip=1,
                screen_size=84,
                terminal_on_life_loss=True,  # 在失去生命时终止回合
                grayscale_obs=True,
                scale_obs=False
            )

            # 帧堆叠
            env = gym.wrappers.FrameStackObservation(env, frame_stack)
            env = Monitor(env)
            return env

        except Exception as e:
            print(f"创建环境 {env_name} 失败: {e}")
            # 回退到Pong
            env = gym.make('ALE/Pong-v5',
                           render_mode='rgb_array',
                           frameskip=4,
                           full_action_space=False)
            env = gym.wrappers.AtariPreprocessing(
                env,
                noop_max=30,
                frame_skip=1,
                screen_size=84,
                terminal_on_life_loss=True,
                grayscale_obs=True,
                scale_obs=False
            )
            env = gym.wrappers.FrameStackObservation(env, frame_stack)
            env = Monitor(env)
            return env

    env = DummyVecEnv([_init])
    return env


def train_improved_atari_agent(env_name='ALE/Pong-v5', total_timesteps=1000000):
    """
    改进的Atari训练函数
    """
    print(f"开始改进训练，环境: {env_name}, 总步数: {total_timesteps}")

    # 创建训练环境
    train_env = create_improved_atari_env(env_name)

    # 改进的策略参数
    policy_kwargs = dict(
        features_extractor_class=ImprovedCNN,
        features_extractor_kwargs=dict(features_dim=512),
        activation_fn=torch.nn.ReLU,
        net_arch=[512, 512]  # 更深的网络
    )

    # 改进的PPO参数
    model = PPO(
        "CnnPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=10,  # 更多的训练轮次
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./improved_atari_tensorboard/",
        verbose=1,
        device='cpu'
    )

    # 改进的训练回调
    class ImprovedProgressCallback(BaseCallback):
        def __init__(self, check_freq: int = 10000, verbose: int = 1):
            super(ImprovedProgressCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.episode_rewards = []
            self.best_mean_reward = -np.inf

        def _on_step(self) -> bool:
            # 记录回合奖励
            if 'episode' in self.locals and 'r' in self.locals['episode']:
                self.episode_rewards.append(self.locals['episode']['r'])

            if self.n_calls % self.check_freq == 0:
                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards[-20:])  # 最近20个回合
                    print(f"步数: {self.num_timesteps}, 最近20回合平均奖励: {mean_reward:.2f}")

                    # 保存最佳模型
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print(f"新的最佳平均奖励: {mean_reward:.2f}, 保存模型...")
                        self.model.save("atari_ppo_best")
                else:
                    print(f"步数: {self.num_timesteps}")

            return True

    # 开始训练
    try:
        print("开始训练...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=ImprovedProgressCallback(check_freq=10000),
            progress_bar=True,
            tb_log_name="Improved_PPO_Atari"
        )

        # 保存最终模型
        model.save("atari_ppo_final")
        print("训练完成！")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        model.save("atari_ppo_interrupted")

    train_env.close()
    return model


def evaluate_improved_agent(model_path=None, env_name='ALE/Pong-v5', n_episodes=10):
    """
    改进的评估函数
    """
    print(f"评估智能体性能，环境: {env_name}")

    # 创建评估环境
    eval_env = create_improved_atari_env(env_name)

    # 加载模型
    if model_path and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print(f"加载模型: {model_path}")
    else:
        try:
            model = PPO.load("atari_ppo_best")  # 尝试加载最佳模型
            print("加载最佳模型")
        except:
            try:
                model = PPO.load("atari_ppo_final")  # 尝试加载最终模型
                print("加载最终模型")
            except:
                print("未找到训练好的模型")
                return None

    # 评估策略
    print(f"评估 {n_episodes} 个回合...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        deterministic=False  # 使用随机策略进行评估，有时效果更好
    )

    print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 详细演示
    print("详细演示运行...")
    demo_env = create_improved_atari_env(env_name)

    total_rewards = []
    for i in range(min(5, n_episodes)):  # 最多5个演示回合
        obs = demo_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 1000:  # 限制最大步数
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = demo_env.step(action)
            total_reward += reward[0]
            steps += 1

            if done:
                break

        total_rewards.append(total_reward)
        print(f"演示回合 {i + 1}: 总奖励={total_reward}, 步数={steps}")

    demo_env.close()
    eval_env.close()

    print(f"演示平均奖励: {np.mean(total_rewards):.2f}")
    return mean_reward


def analyze_training_issue():
    """
    分析训练问题并提供解决方案
    """
    print("\n=== 训练问题分析 ===")
    print("当前问题: 智能体获得0奖励")
    print("\n可能的原因和解决方案:")
    print("1. 训练步数不足 - 增加训练步数到1M或更多")
    print("2. 学习率不合适 - 尝试不同的学习率")
    print("3. 环境配置问题 - 确保环境正确设置")
    print("4. 奖励稀疏 - 考虑使用课程学习或内在好奇心")
    print("5. 网络结构不合适 - 使用更深的网络")

    print("\n推荐的改进措施:")
    print("- 使用更长的训练时间 (1M+ 步数)")
    print("- 使用更深的CNN网络")
    print("- 增加探索 (更高的熵系数)")
    print("- 使用学习率调度器")
    print("- 尝试不同的环境 (如Pong, Breakout)")

    return True


def train_with_curriculum(env_name='ALE/Pong-v5'):
    """
    使用课程学习策略进行训练
    """
    print("使用课程学习策略...")

    # 第一阶段: 基础训练
    print("=== 第一阶段: 基础训练 (250K步) ===")
    model = train_improved_atari_agent(env_name, total_timesteps=250000)

    # 评估第一阶段
    stage1_reward = evaluate_improved_agent("atari_ppo_final", env_name, n_episodes=5)
    print(f"第一阶段平均奖励: {stage1_reward:.2f}")

    # 第二阶段: 精细训练 (如果第一阶段有进步)
    if stage1_reward is not None and stage1_reward > -1:
        print("=== 第二阶段: 精细训练 (250K步) ===")

        # 加载第一阶段模型继续训练
        model = PPO.load("atari_ppo_final")

        # 创建新环境
        train_env = create_improved_atari_env(env_name)

        # 设置模型环境
        model.set_env(train_env)

        # 继续训练
        model.learn(
            total_timesteps=250000,
            progress_bar=True,
            tb_log_name="PPO_Atari_Stage2"
        )

        model.save("atari_ppo_stage2")
        train_env.close()

        # 评估第二阶段
        stage2_reward = evaluate_improved_agent("atari_ppo_stage2", env_name, n_episodes=10)
        print(f"第二阶段平均奖励: {stage2_reward:.2f}")

        improvement = stage2_reward - stage1_reward if stage1_reward is not None else stage2_reward
        print(f"改进: {improvement:.2f}")

    return model


if __name__ == "__main__":
    # 分析问题
    analyze_training_issue()

    # 选择环境
    env_options = ['ALE/Pong-v5', 'ALE/Breakout-v5', 'ALE/SpaceInvaders-v5']

    print("\n可用环境:")
    for i, env in enumerate(env_options):
        print(f"{i + 1}. {env}")

    # 使用Pong环境，因为它通常更容易学习
    selected_env = env_options[0]
    print(f"\n选择环境: {selected_env}")

    # 询问用户训练方式
    print("\n训练选项:")
    print("1. 快速测试 (100K步)")
    print("2. 标准训练 (500K步)")
    print("3. 完整训练 (1M步)")
    print("4. 课程学习 (多阶段训练)")

    choice = 2  # 默认标准训练

    if choice == 1:
        total_steps = 100000
    elif choice == 2:
        total_steps = 500000
    elif choice == 3:
        total_steps = 1000000
    elif choice == 4:
        model = train_with_curriculum(selected_env)
    else:
        total_steps = 500000

    if choice != 4:
        # 单阶段训练
        model = train_improved_atari_agent(selected_env, total_timesteps=total_steps)

        # 评估最终模型
        final_reward = evaluate_improved_agent("atari_ppo_final", selected_env, n_episodes=10)

        # 评估最佳模型
        if os.path.exists("atari_ppo_best.zip"):
            best_reward = evaluate_improved_agent("atari_ppo_best", selected_env, n_episodes=10)
            print(f"最终模型奖励: {final_reward:.2f}, 最佳模型奖励: {best_reward:.2f}")