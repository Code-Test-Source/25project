# 训练完成!
# 评估模型...
# 改进模型评估 - 平均奖励: 348.26 +/- 204.33
# 运行改进模型演示...
# 演示回合 1: 总奖励 = 233.33, 步数 = 1000
# 演示回合 2: 总奖励 = 313.56, 步数 = 1000
# 演示回合 3: 总奖励 = 339.02, 步数 = 1000
# 演示平均奖励: 295.31

import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import torch
import torch.nn as nn


class CarRacingImprovedTraining:
    def __init__(self, total_timesteps=500000):
        self.total_timesteps = total_timesteps
        self.log_dir = "./car_racing_improved/"
        os.makedirs(self.log_dir, exist_ok=True)

    def create_env(self, render_mode="rgb_array"):
        """创建改进的环境"""
        env = gym.make("CarRacing-v3",
                       render_mode=render_mode,
                       continuous=True)  # 使用连续动作空间

        # 包装环境
        env = Monitor(env, self.log_dir)

        return env

    def create_model(self, env):
        """创建改进的PPO模型"""
        # 使用更适合连续控制的超参数
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # 更小的学习率
            n_steps=2048,  # 更多的步数
            batch_size=128,  # 更大的批次
            n_epochs=20,  # 更多的训练轮次
            gamma=0.995,  # 更高的折扣因子
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # 探索系数
            vf_coef=0.5,
            max_grad_norm=0.8,
            policy_kwargs=dict(
                net_arch=[256, 256],  # 更深的网络
                activation_fn=torch.nn.ReLU,
                log_std_init=-0.5,  # 更小的初始标准差
            ),
            tensorboard_log=self.log_dir,
            verbose=1,
            device="auto"
        )

        return model

    def train_with_callbacks(self):
        """使用回调函数进行训练"""
        print("开始改进训练...")

        # 创建训练环境
        train_env = self.create_env()

        # 创建模型
        model = self.create_model(train_env)

        # 创建评估回调
        eval_env = self.create_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.log_dir + "best_model/",
            log_path=self.log_dir + "eval_logs/",
            eval_freq=10000,  # 每10000步评估一次
            deterministic=True,
            render=False
        )

        # 训练进度回调
        class ProgressCallback(BaseCallback):
            def __init__(self, check_freq=1000, verbose=1):
                super(ProgressCallback, self).__init__(verbose)
                self.check_freq = check_freq
                self.episode_rewards = []
                self.current_episode_reward = 0

            def _on_step(self) -> bool:
                # 累积奖励
                self.current_episode_reward += self.locals['rewards'][0]

                # 检查是否回合结束
                if self.locals['dones'][0]:
                    self.episode_rewards.append(self.current_episode_reward)
                    self.current_episode_reward = 0

                    # 定期报告
                    if len(self.episode_rewards) % 10 == 0:
                        mean_reward = np.mean(self.episode_rewards[-10:])
                        print(f"最近10回合平均奖励: {mean_reward:.2f}")

                # 定期报告训练进度
                if self.n_calls % self.check_freq == 0:
                    print(f"训练步数: {self.num_timesteps}")

                return True

        try:
            print("开始第一阶段训练（基础控制）...")
            # 第一阶段：较短训练，建立基础控制能力
            model.learn(
                total_timesteps=100000,
                callback=[ProgressCallback(), eval_callback],
                progress_bar=True
            )

            # 保存第一阶段模型
            model.save(self.log_dir + "car_racing_phase1")

            print("开始第二阶段训练（精细控制）...")
            # 第二阶段：继续训练，学习精细控制
            model.learn(
                total_timesteps=self.total_timesteps - 100000,
                callback=[ProgressCallback(), eval_callback],
                progress_bar=True,
                reset_num_timesteps=False
            )

            # 保存最终模型
            model.save(self.log_dir + "car_racing_final")
            print("训练完成!")

        except KeyboardInterrupt:
            print("训练被中断，保存当前模型...")
            model.save(self.log_dir + "car_racing_interrupted")

        finally:
            train_env.close()
            eval_env.close()

        return model

    def evaluate_improved(self, model_path=None):
        """评估改进的模型"""
        print("评估模型...")

        eval_env = self.create_env()

        # 加载最佳模型
        if model_path:
            model = PPO.load(model_path)
        else:
            try:
                model = PPO.load(self.log_dir + "best_model/best_model")
            except:
                model = PPO.load(self.log_dir + "car_racing_final")

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,  # 更多评估回合
            deterministic=True
        )

        print(f"改进模型评估 - 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

        eval_env.close()
        return mean_reward, std_reward

    def demo_improved(self, model_path=None, episodes=5):
        """演示改进的模型"""
        print("运行改进模型演示...")

        env = self.create_env(render_mode="human")

        # 加载最佳模型
        if model_path:
            model = PPO.load(model_path)
        else:
            try:
                model = PPO.load(self.log_dir + "best_model/best_model")
            except:
                model = PPO.load(self.log_dir + "car_racing_final")

        total_rewards = []

        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < 1000:  # 限制最大步数
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                if done or truncated:
                    break

            total_rewards.append(total_reward)
            print(f"演示回合 {episode + 1}: 总奖励 = {total_reward:.2f}, 步数 = {steps}")

        print(f"演示平均奖励: {np.mean(total_rewards):.2f}")

        env.close()


# 专门针对CarRacing的预训练策略
def pretrain_car_racing():
    """使用预训练策略快速获得正奖励"""
    print("使用预训练策略...")

    # 创建一个简单的启发式策略来获得初始正奖励
    class SimpleHeuristicPolicy:
        def __init__(self):
            self.step_count = 0

        def predict(self, obs, deterministic=True):
            # 简单的启发式策略：轻微向右转并给一点油门
            self.step_count += 1

            # 前几步：给油门直线前进
            if self.step_count < 20:
                return np.array([0.0, 0.3, 0.0]), None
            # 之后：轻微右转
            else:
                return np.array([0.1, 0.2, 0.0]), None

    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)

    # 使用简单策略收集一些经验
    policy = SimpleHeuristicPolicy()
    rewards = []

    for episode in range(5):
        obs, _ = env.reset()
        policy.step_count = 0
        done = False
        total_reward = 0

        while not done and policy.step_count < 200:
            action, _ = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                break

        rewards.append(total_reward)
        print(f"预训练回合 {episode + 1}: 奖励 = {total_reward:.2f}")

    env.close()
    print(f"预训练平均奖励: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    # 首先尝试预训练策略
    print("=== CarRacing 改进训练 ===")

    # 运行预训练演示
    pretrain_car_racing()

    # 创建改进的训练器
    trainer = CarRacingImprovedTraining(total_timesteps=300000)  # 30万步

    # 训练
    model = trainer.train_with_callbacks()

    # 评估
    trainer.evaluate_improved()

    # 演示
    trainer.demo_improved(episodes=3)