import src.EnvGym as EnvGym
import src.EnviromentBullet as EnvBullet
import src.TD3 as Agente
import tensorflow as tf
import gymnasium as gym 
import RegisterEnv
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
import keyboard

# log_dir = "logs/td3/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = tf.summary.create_file_writer(log_dir)

env = EnvGym.EnviromentGymBullet(connect=False)
envs = gym.make("Milimars-v0-Bullet", render_mode="human")

Control_Automatico = Agente.Agent(alpha=0.001, beta=0.001, input_dims=env.player.observation_space.shape,
                  tau=0.005, env=env.player, max_size=10000000, batch_size=256, noise=0.1, warmup=0)
Control_Automatico.load_models()

n_Episodes = 100
Meta = 0

# 📊 Inicializar listas para graficar
vlineal_list = []
vangular_list = []
progress_list = []
alineacion_list = []
reward_list = []

for episode in range(n_Episodes):

    done = False
    score = 0
    cont = 0
    obs_start, _ = envs.reset(options="Random")
    observation = obs_start[4:]
    action = np.zeros(2)
    speed = 0
    rotation = 0
    while not done: # and cont < 1500:

        cont+=1

        #Control Automatico
        # if cont % 1 == 0:
        action = Control_Automatico.choose_action(observation)
        
        # Control Manual Pulsación de tecla
        # if keyboard.is_pressed("up"):
        #     speed = 1
        # elif keyboard.is_pressed("down"):
        #     speed = -1
        # else:
        #     speed = 0

        # if keyboard.is_pressed("left"):
        #     rotation = 1
        # elif keyboard.is_pressed("right"):
        #     rotation = -1
        # else:
        #     rotation = 0

        # action = tf.constant([speed, rotation], dtype=tf.float32)

        new_observation_aux, reward, done, truncated, info = envs.step(action)
        new_observation = new_observation_aux[4:]
        print(new_observation)
        reward = tf.clip_by_value(reward, -10.0, 10.0)

        # Control_Automatico.remember(observation, action, reward, new_observation, done)
        # Control_Automatico.learn()
        score += float(reward)
        observation = new_observation

        # 🔄 Almacenar valores
        vlineal_list.append(float(action[0]))
        vangular_list.append(float(action[1]))
        progress_list.append(float(new_observation_aux[7]))
        alineacion_list.append(float(new_observation_aux[3]))
        reward_list.append(float(reward))

        if cont % 10 == 0:
            print("\rEp:", episode, "Int:", cont, "VLineal:", round(float(action[0]), 4),
            "AngGiro:", round(float(action[1]), 4), "Progress:", round(float(new_observation_aux[7]), 4),
            "Alineación:", round(float(new_observation_aux[3]), 4), "Reward", round(float(reward), 4),
            "Score:", round(score, 4), "Task Complete:", Meta, "     ", end="", flush=True)

        if done == True:
            Meta+=1
            # Control_Automatico.save_models()
            print("EXITO")
            break      

    # 📈 Graficar al final del episodio
    plt.figure(figsize=(12, 8))
    plt.plot(vlineal_list, label="VLineal")
    plt.plot(vangular_list, label="VAngular")
    plt.plot(progress_list, label="Progress")
    plt.plot(alineacion_list, label="Alineación")
    plt.plot(reward_list, label="Reward")
    plt.title(f"Episode {episode} - Metrics")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    vlineal_list.clear()
    vangular_list.clear()
    progress_list.clear()
    alineacion_list.clear()
    reward_list.clear()

    print("Ep", episode, 'score %.2f' % score)
    if Meta == 100:
        break

envs.close()
