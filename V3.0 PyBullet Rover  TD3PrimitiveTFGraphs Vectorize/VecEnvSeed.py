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
        env = gym.make("Milimars-v0-Bullet", disable_env_checker=True)
        env.reset(options="Random") 
        return env
    return thunk

if __name__ == "__main__":


    log_dir = "logs/td3/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1" 


    env = EnvGym.EnviromentGymBullet(connect=False)
    envs = gym.vector.AsyncVectorEnv([make_env() for i in range(5)])


    Control_Automatico = Agente.Agent(alpha=0.001, beta=0.001, input_dims=env.player.observation_space.shape,
                    tau=0.005, env=env.player, max_size=10000000, batch_size=256, noise=0.1)
    # Control_Automatico.load_models()
# 
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
        obs_start = obs_start[:, 4:]
        observation = obs_start
        action = np.zeros((envs.num_envs , 2))

        while not np.any(done) and cont < 3000:
            
            cont+=1 

            #Control Automatico
            action = np.array([Control_Automatico.choose_action(observation[i]) for i in range(envs.num_envs)]) 
            new_observation_aux, reward, done, truncated, infos = envs.step(action)

            reward = tf.clip_by_value(reward, -10.0, 10.0)

            new_observation = new_observation_aux[:, 4:]
            for i in range(envs.num_envs):Control_Automatico.remember(observation[i], action[i], reward[i], new_observation[i], done[i])
            Control_Automatico.learn()

            score += reward
            observation = new_observation

            mean_progress = np.round(np.mean(new_observation_aux[:, 7]), 4)
            mean_alineacion = np.round(np.mean(new_observation_aux[:, 6]), 4)
            mean_reward = np.round(np.mean(reward),4)

            if cont % 25 == 0:
                print("\rEp:", episode, "Int:", cont, "Progress:", mean_progress,
                    "Alineación:", mean_alineacion, "Reward", mean_reward,"Score:", round(np.mean(score),4),
                    "Task Complete:" ,Meta,"     ", end="")

            if np.any(done):
                Meta+=1
                print("EXITO")
                break  


        with writer.as_default():
            tf.summary.scalar(f"ENV/Reward_Means", mean_reward, step=episode)
            tf.summary.scalar(f"ENV/Score_Means", np.mean(score), step=episode)
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
        
        # if episode % 5 == 0:
        #     Control_Automatico.Ajuste_Ruido()
        #     # 📈 Graficar al final del episodio
        #     plt.figure(figsize=(12, 8))
        #     plt.plot(vlineal_list, label="VLineal")
        #     plt.plot(vangular_list, label="VAngular")
        #     plt.plot(progress_list, label="Progress")
        #     plt.plot(alineacion_list, label="Alineación")
        #     plt.plot(reward_list, label="Reward")
        #     plt.title(f"Episode {episode} - Metrics")
        #     plt.xlabel("Step")
        #     plt.ylabel("Value")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()
        
        #     vlineal_list.clear()
        #     vangular_list.clear()
        #     progress_list.clear()
        #     alineacion_list.clear()

        
        if episode % 25 == 0:
            Control_Automatico.save_models()
            Control_Automatico.Ajuste_Ruido()


        if Meta == 100:
            break

    Control_Automatico.save_models()
    envs.close()
