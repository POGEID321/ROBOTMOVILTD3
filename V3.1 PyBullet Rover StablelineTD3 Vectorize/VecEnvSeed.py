from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

import gymnasium as gym
import numpy as np
import os
import datetime
import RegisterEnv  # asegúrate que tu entorno está registrado aquí


def make_env():
    def thunk():
        env = gym.make("Milimars-v0-Bullet", disable_env_checker=True)
        env = Monitor(env)  # para registrar episodios
        return env
    return thunk


if __name__ == "__main__":

    num_envs = 4
    envs = DummyVecEnv([make_env() for _ in range(num_envs)])

    # 📝 Logger para TensorBoard    
    log_dir = f"logs/td3_sb3/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # 🎯 Inicializa modelo TD3
    model = TD3(
        policy="MlpPolicy",
        env=envs,   
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-3,
        batch_size=256,    
        buffer_size=10_000_000,
        train_freq=(4, "step"),
        gradient_steps=2,
        policy_kwargs=dict(net_arch=[256, 256]), 
    )

    # model = TD3.load("td3_milimars_PyBullet10", env=envs)

    model.set_logger(new_logger)

    # 💾 Callback para guardar modelos cada N pasos
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models_sb4/", name_prefix="td3", verbose=1)

    # 🚀 Entrenamiento 
    total_timesteps = 10_000_000  # equivalente a 1500 episodios * ~1000 steps
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # 💾 Guardar modelo final
    model.save("td3_milimars_PyBullet")
    envs.close()
