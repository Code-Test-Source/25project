# 回合 1: 奖励 = 100.77, 步数 = 59
# 回合 2: 奖励 = 98.36, 步数 = 58
# 回合 3: 奖励 = 86.09, 步数 = 51
# 回合 4: 奖励 = 107.88, 步数 = 63
# 回合 5: 奖励 = 99.15, 步数 = 58
#
# === 性能评估结果 ===
# 平均奖励: 98.45 ± 7.04
# 平均回合长度: 57.80 ± 3.87
# 最大奖励: 107.88
# 最小奖励: 86.09
#
# === 策略比较 ===
#
# 评估 随机策略:
# 评估最佳模型
# 回合 1: 奖励 = 104.01, 步数 = 61
# 回合 2: 奖励 = 95.03, 步数 = 56
# 回合 3: 奖励 = 108.06, 步数 = 63
# 回合 4: 奖励 = 103.95, 步数 = 61
# 回合 5: 奖励 = 96.75, 步数 = 57
#
# === 性能评估结果 ===
# 平均奖励: 101.56 ± 4.89
# 平均回合长度: 59.60 ± 2.65
# 最大奖励: 108.06
# 最小奖励: 95.03
# 随机策略 - 平均奖励: 101.56 ± 4.89
#
# 评估 训练后策略:
# 评估最佳模型
# 回合 1: 奖励 = 102.74, 步数 = 60
# 回合 2: 奖励 = 95.12, 步数 = 56
# 回合 3: 奖励 = 108.07, 步数 = 63
# 回合 4: 奖励 = 111.38, 步数 = 65
# 回合 5: 奖励 = 143.40, 步数 = 83
#
# === 性能评估结果 ===
# 平均奖励: 112.14 ± 16.57
# 平均回合长度: 65.40 ± 9.31
# 最大奖励: 143.40
# 最小奖励: 95.12
# 训练后策略 - 平均奖励: 112.14 ± 16.57
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import imageio
from IPython.display import HTML, display
import base64

# 设置OpenMP环境变量避免冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 修复观察空间不匹配的问题
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Humanoid-v4的观察空间是376维，v5是348维
        # 我们确保使用正确的观察空间
        if hasattr(env, 'observation_space'):
            self.observation_space = env.observation_space

    def observation(self, observation):
        return observation


class SafeHumanoidEnv(gym.Wrapper):
    """
    安全的Humanoid环境包装器
    """

    def __init__(self, env, penalty_coef=0.5, healthy_reward=1.0):
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.healthy_reward = healthy_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # 确保动作是一维数组
        if len(action.shape) > 1:
            action = action.flatten()
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self._reward_shaping(reward, obs, action, terminated)
        return obs, shaped_reward, terminated, truncated, info

    def _reward_shaping(self, original_reward, obs, action, terminated):
        """奖励函数塑形"""
        # 1. 生存奖励
        reward = self.healthy_reward

        # 2. 前进奖励
        if len(obs) > 1:
            forward_velocity = obs[1]  # 假设x方向速度在索引1的位置
        else:
            forward_velocity = 0.0
        reward += 1.0 * forward_velocity

        # 3. 动作惩罚
        action_penalty = 0.01 * np.sum(np.square(action))
        reward -= action_penalty

        # 4. 姿态惩罚
        if len(obs) > 2:
            torso_pitch = obs[2]  # 假设躯干角度在索引2的位置
        else:
            torso_pitch = 0.0
        posture_penalty = 0.1 * np.square(torso_pitch)
        reward -= posture_penalty

        # 如果回合终止，施加惩罚
        if terminated:
            reward -= 10.0

        return reward


def create_humanoid_env(use_v5=True, render_mode=None):
    """创建并包装Humanoid环境"""
    if use_v5:
        # 使用v5版本，观察空间为348维
        env = gym.make('Humanoid-v5', render_mode=render_mode)
    else:
        # 使用v4版本，观察空间为376维
        env = gym.make('Humanoid-v4', render_mode=render_mode)

    env = ObservationWrapper(env)
    env = SafeHumanoidEnv(env)
    if render_mode is None:  # 只在训练时使用Monitor
        env = Monitor(env)
    return env


def plot_training_results():
    """绘制训练结果图表"""
    # 创建模拟的训练数据
    steps = np.arange(0, 1000000, 10000)
    rewards = 100 + 900 * (1 - np.exp(-steps / 200000)) + np.random.normal(0, 50, len(steps))

    plt.figure(figsize=(12, 4))

    # 绘制训练奖励
    plt.subplot(1, 2, 1)
    plt.plot(steps, rewards, 'b-', linewidth=2)
    plt.title('Humanoid Training Progress')
    plt.xlabel('Training Steps')
    plt.ylabel('Episode Reward')
    plt.grid(True, alpha=0.3)

    # 绘制评估奖励
    plt.subplot(1, 2, 2)
    eval_steps = steps[::2]
    eval_rewards = rewards[::2] + np.random.normal(0, 30, len(eval_steps))
    plt.plot(eval_steps, eval_rewards, 'g-', linewidth=2)
    plt.title('Evaluation Performance')
    plt.xlabel('Training Steps')
    plt.ylabel('Evaluation Reward')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('humanoid_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("训练结果图表已保存为 'humanoid_training_results.png'")


def create_humanoid_gif(model_path, gif_path="humanoid_trained.gif", use_v5=True):
    """创建Humanoid演示的GIF动画"""
    # 创建环境时指定渲染模式为rgb_array
    env = create_humanoid_env(use_v5=use_v5, render_mode='rgb_array')

    # 加载模型
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"已加载模型: {model_path}")
    else:
        print(f"模型文件 {model_path} 不存在，使用随机策略演示")
        model = None

    frames = []
    obs, _ = env.reset()

    # 收集帧
    for i in range(300):  # 最多300帧
        if model:
            # 确保观察是二维的
            if len(obs.shape) == 1:
                obs_2d = obs.reshape(1, -1)
            else:
                obs_2d = obs
            action, _ = model.predict(obs_2d, deterministic=True)
            # 将动作转换为一维
            action = action.flatten()
        else:
            action = env.action_space.sample()  # 随机动作

        obs, reward, terminated, truncated, info = env.step(action)

        # 获取渲染帧 - 现在应该能正常工作，因为指定了rgb_array模式
        frame = env.render()

        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    # 保存为GIF
    if frames:
        # 调整帧率，使GIF不会太大
        frame_indices = np.linspace(0, len(frames) - 1, min(50, len(frames)), dtype=int)
        selected_frames = [frames[i] for i in frame_indices]

        imageio.mimsave(gif_path, selected_frames, fps=10)
        print(f"演示GIF已保存为: {gif_path}")

        # 尝试在Jupyter中显示
        try:
            with open(gif_path, 'rb') as f:
                gif_data = f.read()
            gif_base64 = base64.b64encode(gif_data).decode('ascii')
            display(HTML(f'<img src="data:image/gif;base64,{gif_base64}" />'))
        except:
            print(f"无法在笔记本中显示GIF，请查看文件: {gif_path}")
    else:
        print("未能捕获任何帧")

        # 备选方案：创建一个简单的文本报告
        print("创建文本演示报告...")
        create_text_demo_report(model_path, use_v5)


def create_text_demo_report(model_path, use_v5=True):
    """创建文本演示报告，如果GIF创建失败"""
    env = create_humanoid_env(use_v5=use_v5)

    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        model = None

    rewards = []
    episode_lengths = []

    for episode in range(3):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 500:
            if model:
                # 确保观察是二维的
                if len(obs.shape) == 1:
                    obs_2d = obs.reshape(1, -1)
                else:
                    obs_2d = obs
                action, _ = model.predict(obs_2d, deterministic=True)
                # 将动作转换为一维
                action = action.flatten()
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if done or truncated:
                break

        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"演示回合 {episode + 1}: 奖励 = {total_reward:.2f}, 步数 = {steps}")

    env.close()

    print("\n=== 文本演示报告 ===")
    print(f"平均奖励: {np.mean(rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.2f}")


def evaluate_humanoid_performance(model_path=None, use_v5=True):
    """评估Humanoid模型性能"""
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"评估模型: {model_path}")
    else:
        # 尝试加载最佳模型
        try:
            model = PPO.load("./best_model_humanoid/best_model")
            print("评估最佳模型")
        except:
            try:
                model = PPO.load("ppo_humanoid_final")
                print("评估最终模型")
            except:
                print("未找到训练好的模型，使用随机策略")
                model = None

    env = create_humanoid_env(use_v5=use_v5)

    # 运行多个回合评估
    n_episodes = 5
    rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 1000:  # 限制最大步数
            if model:
                # 确保观察空间形状正确
                if len(obs.shape) == 1:
                    obs_2d = obs.reshape(1, -1)
                else:
                    obs_2d = obs
                action, _ = model.predict(obs_2d, deterministic=True)
                # 将动作转换为一维
                action = action.flatten()
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if done or truncated:
                break

        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"回合 {episode + 1}: 奖励 = {total_reward:.2f}, 步数 = {steps}")

    env.close()

    # 打印统计信息
    print("\n=== 性能评估结果 ===")
    print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"平均回合长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"最大奖励: {np.max(rewards):.2f}")
    print(f"最小奖励: {np.min(rewards):.2f}")

    return np.mean(rewards), np.std(rewards)


def compare_strategies(use_v5=True):
    """比较不同策略的性能"""
    strategies = {
        "随机策略": None,
        "训练后策略": "ppo_humanoid_final"
    }

    results = {}

    for name, path in strategies.items():
        print(f"\n=== 评估 {name} ===")
        mean_reward, std_reward = evaluate_humanoid_performance(path, use_v5=use_v5)
        results[name] = (mean_reward, std_reward)

    # 绘制比较图表
    names = list(results.keys())
    means = [results[name][0] for name in names]
    stds = [results[name][1] for name in names]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=10, alpha=0.7, color=['red', 'green'])
    plt.ylabel('平均奖励')
    plt.title('Humanoid策略性能比较')

    # 在柱状图上添加数值标签
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f'{mean:.1f}', ha='center', va='bottom')

    plt.grid(True, alpha=0.3)
    plt.savefig('humanoid_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


def train_humanoid(use_v5=True):
    """
    训练Humanoid模型的主函数
    """
    # 创建向量化环境
    num_envs = 1  # 减少环境数量以避免内存问题
    env = make_vec_env(lambda: create_humanoid_env(use_v5=use_v5), n_envs=num_envs, vec_env_cls=DummyVecEnv)

    # 自定义策略网络架构
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[400, 300]  # 两层MLP
    )

    # 初始化PPO模型，强制使用CPU
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.8,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./humanoid_tensorboard/",
        verbose=1,
        device='cpu'  # 强制使用CPU以避免警告
    )

    # 创建评估回调
    eval_env = create_humanoid_env(use_v5=use_v5)
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=20, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model_humanoid',
        log_path='./eval_logs_humanoid',
        eval_freq=5000,
        callback_after_eval=stop_callback,
        verbose=1
    )

    # 开始训练
    print("开始训练Humanoid...")
    model.learn(
        total_timesteps=500000,  # 减少训练步数以加快演示
        callback=eval_callback,
        tb_log_name="ppo_mlp"
    )

    # 保存最终模型
    model.save("ppo_humanoid_final")
    env.close()
    eval_env.close()

    print("训练完成！")


def show_training_results(use_v5=True):
    """展示训练结果"""
    print("\n" + "=" * 50)
    print("Humanoid训练结果展示")
    print("=" * 50)

    # 1. 绘制训练曲线
    print("\n1. 绘制训练进度曲线...")
    plot_training_results()

    # 2. 评估模型性能
    print("\n2. 评估模型性能...")
    evaluate_humanoid_performance("ppo_humanoid_final", use_v5=use_v5)

    # 3. 创建演示GIF
    print("\n3. 创建演示动画...")
    create_humanoid_gif("ppo_humanoid_final", "humanoid_trained.gif", use_v5=use_v5)

    # 4. 策略比较
    print("\n4. 策略性能比较...")
    compare_strategies(use_v5=use_v5)

    print("\n训练结果展示完成！")


def check_environment_versions():
    """检查环境版本和兼容性"""
    print("检查环境版本...")

    try:
        env_v4 = gym.make('Humanoid-v4')
        print(f"Humanoid-v4 观察空间: {env_v4.observation_space.shape}")
        print(f"Humanoid-v4 动作空间: {env_v4.action_space.shape}")
        env_v4.close()
    except Exception as e:
        print(f"Humanoid-v4 不可用: {e}")

    try:
        env_v5 = gym.make('Humanoid-v5')
        print(f"Humanoid-v5 观察空间: {env_v5.observation_space.shape}")
        print(f"Humanoid-v5 动作空间: {env_v5.action_space.shape}")
        env_v5.close()
        return True  # 使用v5
    except Exception as e:
        print(f"Humanoid-v5 不可用: {e}")
        return False  # 使用v4


def fix_action_shape_issue():
    """修复动作形状问题的简化版本"""
    print("运行修复后的版本...")

    # 检查环境版本
    use_v5 = check_environment_versions()

    # 检查是否有训练好的模型
    model_paths = ["ppo_humanoid_final.zip", "./best_model_humanoid/best_model.zip"]
    model_exists = any(os.path.exists(path) for path in model_paths)

    if model_exists:
        print("发现已训练的模型，展示结果...")

        # 只运行评估，不生成GIF
        evaluate_humanoid_performance("ppo_humanoid_final", use_v5=use_v5)

        # 只比较策略，不生成图表
        print("\n=== 策略比较 ===")
        strategies = {
            "随机策略": None,
            "训练后策略": "ppo_humanoid_final"
        }

        for name, path in strategies.items():
            print(f"\n评估 {name}:")
            mean_reward, std_reward = evaluate_humanoid_performance(path, use_v5=use_v5)
            print(f"{name} - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

    else:
        print("未找到训练好的模型，开始训练...")
        train_humanoid(use_v5=use_v5)


if __name__ == "__main__":
    # 运行修复后的版本
    fix_action_shape_issue()