import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from . import EnviromentBullet as Env 
import math
import pybullet as p
import time

class EnviromentGymBullet(gym.Env):

    def __init__(self, render_mode = None, connect=True):
        super().__init__()

        # Solo inicializar si es renderizable
        if connect:
            if render_mode == "human":
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)

        # Crear Enviroment
        self.player = Env.Player()
        self.goal = Env.Goal()

        self.obstacles = []
        self.Pgoal = [[0, 2], [2, 2], [2, 0], [2, -2], [0, -2], [-2, -2], [-2, 0], [-2, 2]]
        self.Pplayer = [self.player.x,self.player.y]
        self.xstart = 0
        self.ystart = 0
        self.interaction = 0
        self.action_prev = 0

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=self.player.action_space_low, high=self.player.action_space_high, dtype=np.float32)

    def get_observation(self):

        xplayer, yplayer = self.player.update_location()
        xgoal, ygoal = self.goal.update_location()
        angle = math.degrees(self.player.angle)

        Vel = self.player.speed
        Angwheel = self.player.angwheel

        Velx = Vel * math.cos(self.player.angle+math.pi)
        Vely = Vel * math.sin(self.player.angle+math.pi)


        Agoal = math.atan2(ygoal-yplayer, xgoal-xplayer) 
        Alineacion = math.cos(Agoal- (angle+180) * math.pi/180)
        
        DesfaseAngular =  math.sin(((Agoal - (angle+180) * math.pi/180 + math.pi)  % (2 * math.pi) - math.pi)/ 2)
        
        if self.interaction != 0:

            if (xgoal - self.xstart) != 0:
                ErrX = (xgoal - xplayer) / (xgoal - self.xstart)
            else: ErrX = xgoal - xplayer
            if (ygoal - self.ystart) != 0:
                ErrY = (ygoal - yplayer) / (ygoal - self.ystart)  
            else: ErrY = ygoal - yplayer

            #Calculo de progreso de recorrido
            initial_distance = np.linalg.norm([xgoal - self.xstart, ygoal - self.ystart])
            actual_distance = np.linalg.norm([xgoal - xplayer, ygoal - yplayer])
            Progreso = (initial_distance - actual_distance)/initial_distance
        
        else:

            ErrX = 0
            ErrY = 0
            Progreso = 0

        ErrX = np.clip(ErrX, -1.0, 1.0)
        ErrY = np.clip(ErrY, -1.0, 1.0)
        Progreso = np.clip(Progreso, -1.0, 1.0)


        return np.array([round(xplayer,4),round(yplayer,4),
                        round(float(angle),4),round(Alineacion,4),
                        # round(float(Velx),4),round(float(Vely),4),
                        # round(float(ErrX),4),round(float(ErrY),4),
                        round(float(Vel),4),round(float(Angwheel),4),
                        round(DesfaseAngular,4),round(Progreso,4)])

    def reset(self, seed=None, options=None):


        super().reset(seed=seed)

        if options == "Stable":
            
            self.np_random, _ = seeding.np_random(seed)
            idx = self.np_random.integers(low=0, high=8) 

            self.player.locate(self.Pplayer[0],self.Pplayer[1])
            self.goal.locate(self.Pgoal[idx][0],self.Pgoal[idx][1])

        if options == "Random":

            self.np_randompx, _ = seeding.np_random(seed)
            self.np_randompy, _ = seeding.np_random(seed)
            self.np_randomgx, _ = seeding.np_random(seed)
            self.np_randomgy, _ = seeding.np_random(seed)
            px = py = gx = gy = 0

            while np.linalg.norm([gx - px, gy - py]) <= 4 or np.linalg.norm([gx - px, gy - py]) >= 6:
            
                px = self.np_randompx.integers(low=-Env.WIDTH/2, high=(Env.HEIGHT/2)+1) 
                py = self.np_randompy.integers(low=-Env.WIDTH/2, high=(Env.HEIGHT/2)+1) 
                
                gx = self.np_randomgx.integers(low=-Env.WIDTH/2, high=(Env.HEIGHT/2)+1) 
                gy = self.np_randomgy.integers(low=-Env.WIDTH/2, high=(Env.HEIGHT/2)+1) 

            self.player.locate(px,py)
            self.goal.locate(gx,gy)

        info = {} 

        self.interaction = 0
        self.xstart = self.get_observation()[0]
        self.ystart = self.get_observation()[1]

        return self.get_observation(), info 


    def step(self, action):
        self.interaction += 1

        # Aplicar acción
        velocity, angular_velocity = action
        angular_velocity = np.array(angular_velocity).astype(np.float32)

        self.player.move(velocity, angular_velocity)
        
        p.stepSimulation()

        # Obtener observación
        new_observation_space = self.get_observation()

        xplayer, yplayer = new_observation_space[0], new_observation_space[1]
        xgoal, ygoal = self.goal.update_location()

        # - Calcula Recompensa 
        reward = 0.5 * (action[0]**2) + 8 * new_observation_space[7] + 2 * new_observation_space[3] # + 10 * (0.1*new_observation_space[6] * new_observation_space[5]) 

        done = False
        
        #Calculo de progreso de recorrido
        actual_distance = np.linalg.norm([xgoal - xplayer, ygoal - yplayer])
        if abs(actual_distance) < 0.1:
            reward += 2
            done = True

        truncated = False
        info = {}

        return new_observation_space, reward, done, truncated, info  # Devuelve el nuevo estado, la recompensa, si el episodio ha terminado y la información adicional

    def close(self):
        p.disconnect()