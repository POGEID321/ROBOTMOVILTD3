import src.EnvGym as EnvGym
import src.TD3 as Agente
import tensorflow as tf
import gymnasium as gym 
import RegisterEnv
import numpy as np

envs = gym.make_vec("Milimars-v0", num_envs=5, vectorization_mode="sync",seeds=[0, 1, 2, 3, 4])
env = EnvGym.EnviromentGym()

Control_Automatico = Agente.Agent(alpha=0.001, beta=0.001, input_dims=env.player.observation_space.shape,
                  tau=0.005, env=env.player, batch_size=64, noise=0.1)

n_Episodes = 500
Meta = 0

for episode in range(n_Episodes):

    done = False
    cont = 0
    score = 0
    obs_start, _ = envs.reset()
    obs_start = obs_start[:, 3:]
    observation = obs_start
    while not np.any(done) and cont < 300:

        cont+=1
        print("Ep:",episode,",Int:",cont,"Task Complete:",Meta)

        #Control Automatico
        action = np.array([Control_Automatico.choose_action(observation[i]) for i in range(envs.num_envs)])  # Control_Automatico.choose_action(observation) 

        new_observation, reward, done, truncated, info = envs.step(action)  # envs.step(action)
        reward = tf.clip_by_value(reward, -10.0, 10.0)
        new_observation = new_observation[:, 3:]
        for i in range(envs.num_envs):Control_Automatico.remember(observation[i], action[i], reward[i], new_observation[i], done[i])
        Control_Automatico.learn()
        score += reward
        observation = new_observation
        
        if np.any(done):
            Meta+=1
            print("EXITO")
            break

        envs.render()      

    print("Ep", episode, 'score %.2f', score)
    if Meta == 100:
        break

envs.close()
