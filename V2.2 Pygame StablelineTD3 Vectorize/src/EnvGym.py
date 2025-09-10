import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from . import Enviroment as Env 
import math
import pygame
import time

class EnviromentGym(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode = None):
        super().__init__()
        
        self.render_mode = render_mode

        # Crear Enviroment
        self.player = Env.Player()
        self.goal = Env.Goal()
        self.Scene = Env.pygame.sprite.Group(self.player,self.goal) #, *obstacles)

        self.clock = pygame.time.Clock()

        # Solo inicializar Pygame si es renderizable
        if render_mode == "human":
            self.screen = pygame.display.set_mode((Env.WIDTH, Env.HEIGHT))
        else:
            self.screen = None

        self.obstacles = []
        self.Pgoal = [[0, 100], [100, 100], [100, 0], [100, -100], [0, -100], [-100, -100], [-100, 0], [-100, 100]]
        self.Pplayer = [self.player.x,self.player.y]
        self.xtart = 0
        self.ytart = 0
        self.interaction = 0
        self.maxinteractions = 500
        self.action_prev = [0,0]
        self.progress_prev = 0

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=self.player.action_space_low, high=self.player.action_space_high, dtype=np.float32)


    def get_observation(self):

        xplayer, yplayer = self.player.update_location()
        xgoal, ygoal = self.goal.update_location()
        angle= math.radians(self.player.angle)

        Velx = self.player.speed * math.cos(angle+math.pi/2)
        Vely = self.player.speed * math.sin(angle+math.pi/2)

        Agoal = math.atan2(ygoal-yplayer, xgoal-xplayer) 
        Alineacion = math.cos(Agoal-(self.player.angle+90)*math.pi/180)
        DesfaseAngular =  math.sin(((Agoal - (self.player.angle+90) * math.pi/180 + math.pi)  % (2 * math.pi) - math.pi)/ 2)

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

        return np.array([round(xplayer/100,4),round(yplayer/100,4),
                        round(float(angle)/(math.pi/2),4),round(Alineacion,4),
                        round(float(Velx),4),round(float(Vely),4),
                        round(ErrX/100,4),round(ErrY/100,4),
                        round(DesfaseAngular,4),round(Progreso,4)])


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.np_randompx, _ = seeding.np_random(seed)
        self.np_randompy, _ = seeding.np_random(seed)
        self.np_randomgx, _ = seeding.np_random(seed)
        self.np_randomgy, _ = seeding.np_random(seed)
        px = py = gx = gy = 0

        while np.linalg.norm([gx - px, gy - py]) <= 800 or np.linalg.norm([gx - px, gy - py]) >= 1200:
        
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

        return self.get_observation()[4:], info 


    def step(self, action):
        self.interaction += 1

        # Aplicar acción
        velocity, angular_velocity = action
        angular_velocity = np.array(angular_velocity).astype(np.float32)
        self.player.move(velocity, angular_velocity)

        self.Scene.update()

        # Obtener observación
        new_observation_space = self.get_observation()

        # - Calcula Recompensa 
        Mov = action[0]**2
        reward =  10 * new_observation_space[3] + 8 * (new_observation_space[9] - self.progress_prev) + 4 * (action[0] - self.action_prev[0])
        reward -= 0.001 * self.interaction

        self.progress_prev = new_observation_space[9]
        self.action_prev = action
        
        if new_observation_space[9] > 0.9:
            reward += 10
            done = True
        else:
            done = False

        if self.interaction >= self.maxinteractions:
            truncated = True
        else:
            truncated = False

        info = {}

        return new_observation_space[4:], reward, done, truncated, info  # Devuelve el nuevo estado, la recompensa, si el episodio ha terminado y la información adicional

    def render(self):

        if self.render_mode != "human" :
            return

        for event in Env.pygame.event.get():
            if event.type == Env.pygame.QUIT:
                Env.pygame.quit()

        # Actualizar y dibujar
        self.screen.fill(Env.BLACK)
        self.Scene.draw(self.screen)
        Env.pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        Env.pygame.quit()