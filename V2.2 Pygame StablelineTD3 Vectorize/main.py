from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import time
import RegisterEnv  # Asegúrate de registrar tu entorno

# Crear entorno con renderizado
env = gym.make("Milimars-v0", render_mode="human", disable_env_checker=True)
env = Monitor(env)

# Cargar el modelo guardado
model = TD3.load("td3_milimars_final2", env=env)

# Ejecutar episodios de prueba
n_test_episodes = 100

for ep in range(n_test_episodes):
    obs, _ = env.reset(options="Random")
    done = False
    total_reward = 0
    step = 0

    while not done and step < 500:
        
        if step % 1 == 0: action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        env.render()

    print(f"✅ Episodio {ep+1} finalizado | Recompensa total: {total_reward:.2f} | Pasos: {step}")

env.close()
