import tensorflow as tf
import gymnasium as gym 
import RegisterEnv
import numpy as np
import keyboard
# from src import ControlGamesir
import math

# Control = ControlGamesir.GameSirController()
env = gym.make("Milimars-v0-Bullet", render_mode="human", disable_env_checker=True)

n_Episodes = 100
Meta = 0

for i in range(n_Episodes):

    done = False
    truncated = False
    cont = 0
    score = 0
    obs_start, _ = env.reset(options="Random")
    observation = obs_start[3:]
    action = np.zeros(2)

    while not done: #and not truncated:

        cont+=1

        # Control Manual
        if keyboard.is_pressed("up"):
            speed = 1
        elif keyboard.is_pressed("down"):
            speed = -1
        else:
            speed = 0

        if keyboard.is_pressed("left"):
            rotation = 1
        elif keyboard.is_pressed("right"):
            rotation = -1
        else:
            rotation = 0

        # estado = Control.read_state()
        # J1x = estado["sticks"]["LX"]
        # J1y = estado["sticks"]["LY"] * -1
        # RT =  estado["triggers"]["RT"]
        # LT =  estado["triggers"]["LT"]

        # speed = RT - LT
        # if(J1x == 0 and J1y == 0): rotation = 0
        # else: rotation = max(min(math.cos(math.atan2(J1y, J1x)),math.cos(math.pi/4)),math.cos(3*math.pi/4)) * -math.sqrt(2)

        action = tf.constant([speed, rotation], dtype=tf.float32)

        new_observation, reward, done, truncated, info = env.step(action)
        observation = new_observation
        
        print("\rReward", round(float(reward),4), np.round(observation,4), end="                 ", flush=True)
        
        if done == True:
            Meta+=1
            print("EXITO")
            break

    if Meta == 100:
        break

env.close()
