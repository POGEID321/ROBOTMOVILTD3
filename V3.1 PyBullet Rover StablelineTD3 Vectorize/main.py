from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import time
import RegisterEnv  # Asegúrate de registrar tu entorno
import keyboard

# Crear entorno con renderizado
env = gym.make("Milimars-v0-Bullet", render_mode="human", disable_env_checker=True)
env = Monitor(env)

# Cargar el modelo guardado
model = TD3.load("td3_milimars_PyBullet", env=env)

# Ejecutar episodios de prueba
n_test_episodes = 100

for ep in range(n_test_episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0

    while not done and not truncated:
        
        if step % 5 == 0: action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        print("\r", round(reward,4), obs, flush=True)
    print(f"✅ Episodio {ep+1} finalizado | Recompensa total: {total_reward:.2f} | Pasos: {step}")

env.close()