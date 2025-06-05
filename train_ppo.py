import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from scalping_env import ScalpingBinanceEnv

def make_env():
    def _init():
        env = ScalpingBinanceEnv()
        return env
    return _init

def train_ppo(total_timesteps=200_000):
    env = DummyVecEnv([make_env()])

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path='./models/', name_prefix='ppo_scalping')

    model = PPO('MlpPolicy', env, verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log="./tensorboard_scalping/")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("ppo_scalping_final")
    env.close()

def test_agent(model_path="ppo_scalping_final", episodes=5):
    env = ScalpingBinanceEnv()
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episódio {ep+1}: Recompensa total = {total_reward}")

if __name__ == "__main__":
    train_ppo()
    test_agent()
