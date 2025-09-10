import src.EnvGym as EnvGym
import src.TD3 as Agente
import tensorflow as tf
import gymnasium as gym 
import RegisterEnv
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
import os


def make_env():
    def thunk():
        env = gym.make("Milimars-v0", disable_env_checker=True)
        env.reset(options="Random") 
        return env
    return thunk

if __name__ == "__main__":

    log_dir = "logs/td3/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir)

    envs = gym.vector.AsyncVectorEnv([make_env() for i in range(12)])
    env = EnvGym.EnviromentGym()

    Control_Automatico = Agente.Agent(alpha=0.001, beta=0.001, input_dims=env.player.observation_space.shape,
                    tau=0.005, env=env.player, max_size=10000000, batch_size=64, noise=0.1)
    # Control_Automatico.load_models()

    n_Episodes = 1500
    Meta = 0

    # 📊 Inicializar listas para graficar
    # vlineal_list = []
    # vangular_list = []
    # progress_list = []
    # alineacion_list = []
    # reward_list = []

    for episode in range(n_Episodes):

        time_start = time.time()
        done = False
        cont = 0
        score = 0
        obs_start, _ = envs.reset(options="Random")
        # print(obs_start)
        observation = obs_start
        action = np.zeros((envs.num_envs , 2))
        
        while not np.any(done) and cont < 1600:

            #Control Automatico
            action = np.array([Control_Automatico.choose_action(observation[i]) for i in range(envs.num_envs)])  # Control_Automatico.choose_action(observation) 

            new_observation, reward, done, truncated, info = envs.step(action)  # envs.step(action)

            for i in range(envs.num_envs): Control_Automatico.remember(observation[i], action[i], reward[i], new_observation[i], done[i])

            if cont % 4 == 0: Control_Automatico.learn()
            score += reward
            observation = new_observation

            mean_alineacion = np.round(np.mean(new_observation[:, 4]), 4)
            mean_progress = np.round(np.mean(new_observation[:, 5]), 4)
            mean_reward = round(np.mean(reward),4)
            mean_score = round(np.mean(score),4)

            if cont % 20 == 0:
                print("\rEp:", episode, "Int:", cont, "Mean Progress:", mean_progress,
                    "Alineación:", mean_alineacion,"Reward",mean_reward,
                    "Score:", mean_score,"Task Complete:" ,Meta, end="")

            if np.any(done):
                Meta+=1
                print("EXITO")
                break

            envs.render()      
            cont+=1

        print("TEpisode:",  time.time() - time_start)

        with writer.as_default():
            tf.summary.scalar(f"ENV/Reward_Means", mean_reward, step=episode)
            tf.summary.scalar(f"ENV/Score_Means", mean_score, step=episode)
            tf.summary.scalar(f"ENV/Alineacion_Means", mean_alineacion, step=episode)
            tf.summary.scalar(f"ENV/Recorrido_Means", mean_progress, step=episode)
            tf.summary.scalar("TD3/Actor_Loss", Control_Automatico.actor_loss, step=episode)
            tf.summary.scalar("TD3/Critic1_Loss", Control_Automatico.critic_1_loss, step=episode)
            tf.summary.scalar("TD3/Critic2_Loss", Control_Automatico.critic_2_loss, step=episode)
            tf.summary.scalar("TD3/Q Value", Control_Automatico.q_value, step=episode)
            for i, weight in enumerate(Control_Automatico.actor.trainable_variables):
                tf.summary.histogram(f"actor/layer_{i}_{weight.name}", weight, step=episode)
            for i, weight in enumerate(Control_Automatico.critic_1.trainable_variables):
                tf.summary.histogram(f"critic_1/layer_{i}_{weight.name}", weight, step=episode)
            for i, weight in enumerate(Control_Automatico.critic_2.trainable_variables):
                tf.summary.histogram(f"critic_2/layer_{i}_{weight.name}", weight, step=episode)
            writer.flush()

        Control_Automatico.Ajuste_Ruido()

        if episode % 25 == 0:
            Control_Automatico.save_models()

        if Meta == 100:
            break

    Control_Automatico.save_models()
    envs.close()
