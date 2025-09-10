import src.EnvGym as EnvGym
import src.TD3 as Agente
import tensorflow as tf
import gymnasium as gym 
import RegisterEnv
import pygame
import numpy as np

env = EnvGym.EnviromentGym()
envs = gym.make("Milimars-v0", render_mode="human")


Control_Automatico = Agente.Agent(alpha=0.001, beta=0.001, input_dims=env.player.observation_space.shape,
                  tau=0.005, env=env.player, batch_size=64, noise=0.1, warmup=0)
Control_Automatico.load_models()
n_Episodes = 4
Meta = 0

for i in range(n_Episodes):

    done = False
    cont = 0
    score = 0
    obs_start, _ = envs.reset(options="Random")
    observation = obs_start[3:]
    action = np.zeros(2)

    while not done and cont < 1000:

        cont+=1
        print("Ep:",i,",Int:",cont,"Task Complete:",Meta)

        #Control Automatico
        if cont % 5 == 0:
            action = Control_Automatico.choose_action(observation)

        # Control Manual
        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_a]:
        #     rotation = 5
        # elif keys[pygame.K_d]:
        #     rotation = -5
        # elif keys[pygame.K_w]:
        #     speed = 10
        # elif keys[pygame.K_s]:
        #     speed = -10
        # else:
        #     rotation = 0
        #     speed = 0
        # action = tf.constant([speed, rotation], dtype=tf.float32)

        new_observation, reward, done, truncated, info = envs.step(action)
        reward = tf.clip_by_value(reward, -10.0, 10.0)
        new_observation = new_observation[3:]
        Control_Automatico.remember(observation, action, reward, new_observation, done)
        Control_Automatico.learn()
        score += reward.numpy().item()
        observation = new_observation
        
        if done == True:
            Meta+=1
            print("EXITO")
            break

        envs.render()      

    Control_Automatico.Ajuste_Ruido()
    print("Ep", i, 'score %.2f' % score)
    if Meta == 100:
        break

Control_Automatico.save_models()
envs.close()
