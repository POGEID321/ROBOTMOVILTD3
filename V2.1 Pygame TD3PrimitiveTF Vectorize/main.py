import src.EnvGym as EnvGym
import src.TD3 as Agente
import tensorflow as tf
import gymnasium as gym 
import RegisterEnv
import pygame
import keyboard
import numpy as np
import datetime
import matplotlib.pyplot as plt

log_dir = "logs/td3/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

envs = gym.make("Milimars-v0", render_mode="human")
env = EnvGym.EnviromentGym()

Control_Automatico = Agente.Agent(alpha=0.001, beta=0.001, input_dims=env.player.observation_space.shape,
                  tau=0.005, env=env.player, batch_size=64, noise=0.1, warmup=0)
# Control_Automatico.load_models()

n_Episodes = 100
Meta = 0

# 📊 Inicializar listas para graficar
vlineal_list = []
vangular_list = []
progress_list = []
alineacion_list = []

for i in range(n_Episodes):

    done = False
    score = 0
    cont = 0
    obs_start, _ = envs.reset(options="Random")
    observation = obs_start
    action = np.zeros(2)

    while not done and cont < 12000: 

        cont+=1

        #Control Automatico
        # if cont % 5 == 0:
        # action = Control_Automatico.choose_action(observation)

        if keyboard.is_pressed("up"):
            speed = 10
        elif keyboard.is_pressed("down"):
            speed = -10
        else:
            speed = 0

        if keyboard.is_pressed("left"):
            rotation = 5
        elif keyboard.is_pressed("right"):
            rotation = -5
        else:
            rotation = 0

        action = tf.constant([speed, rotation], dtype=tf.float32)

        new_observation, reward, done, truncated, info = envs.step(action)
        score += float(reward)
        observation = new_observation
        print(observation)
        # 🔄 Almacenar valores
        vlineal_list.append(float(action[0]))
        vangular_list.append(float(action[1]))
        progress_list.append(float(new_observation[5]))
        alineacion_list.append(float(new_observation[4]))

        if cont % 10 == 0:
            print("\rEp:", i, "Int:", cont, "VLineal:", round(float(action[0]), 4),
                "VAngular", round(float(action[1]), 4), "Progress:", round(float(new_observation[5]), 4),
                "Alineación:", np.round(float(new_observation[4]), 4), "Reward", round(float(reward), 4),
                "Score:", round(score, 4), "Task Complete:", Meta, end="", flush=True)

        if done == True:
            Meta+=1
            print("EXITO")
            break

        envs.render()      

    # 📈 Graficar al final del episodio
    plt.figure(figsize=(12, 8))
    plt.plot(vlineal_list, label="VLineal")
    plt.plot(vangular_list, label="VAngular")
    plt.plot(progress_list, label="Progress")
    plt.plot(alineacion_list, label="Alineación")
    plt.title(f"Episode {i} - Metrics")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Ep", i, 'score %.2f' % score)
    if Meta == 100:
        break

envs.close()
