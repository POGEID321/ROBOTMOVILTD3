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
        self.maxinteractions = 2000
        self.action_prev = [0,0]
        self.progress_prev_position = 1

        # --- Configuración del LIDAR ---
        self.ray_length = 10
        self.angles = np.linspace(np.pi/2, 3*np.pi/2, 36, endpoint=True)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=self.player.action_space_low, high=self.player.action_space_high, dtype=np.float32)

    def get_observation_vision(self):

        sensor_pos, sensor_orn = p.getLinkState(self.player.Robot, 14)[0:2]

        # Generar rayos en marco local del sensor
        ray_from, ray_to = [], []
        for angle in self.angles:
            dx = self.ray_length * np.cos(angle)
            dy = self.ray_length * np.sin(angle)

            local_from = [0, 0, 0]  # origen del rayo en el link
            local_to   = [dx,dy,0]  # dirección en XY

            # Transformar al mundo
            world_from, _ = p.multiplyTransforms(sensor_pos, sensor_orn, local_from, [0,0,0,1])
            world_to, _   = p.multiplyTransforms(sensor_pos, sensor_orn, local_to, [0,0,0,1])

            ray_from.append(world_from)
            ray_to.append(world_to)

        # Lanzar todos los rayos
        results = p.rayTestBatch(ray_from, ray_to)
        p.removeAllUserDebugItems()
        # Procesar distancias
        distances = []
        for r in results:
            hit_uid, hit_fraction = r[0], r[2]
            dist = hit_fraction * self.ray_length if hit_uid != -1 else self.ray_length
            distances.append(dist)

        for i, r in enumerate(results):
            color = [1, 0, 0] if r[0] != -1 else [0, 1, 0]  # rojo si impacta, verde si libre
            p.addUserDebugLine(ray_from[i], ray_to[i], lineColorRGB=color, lineWidth=1, lifeTime=0.05)

        return distances

    def get_observation_position(self):

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
                        # round(float(Vel),4),round(float(Angwheel),4),
                        round(DesfaseAngular,4),round(Progreso,4)])

    def reset(self, seed=None, options=None):


        super().reset(seed=seed)

        self.rng = np.random.default_rng(seed)
        px = py = gx = gy = 0

        while np.linalg.norm([gx - px, gy - py]) <= 2 or np.linalg.norm([gx - px, gy - py]) >= 5:
        
            px = self.rng.integers(low=-Env.WIDTH/2, high=(Env.HEIGHT/2)+1) 
            py = self.rng.integers(low=-Env.WIDTH/2, high=(Env.HEIGHT/2)+1) 
            
            gx = self.rng.integers(low=-Env.WIDTH/2, high=(Env.HEIGHT/2)+1) 
            gy = self.rng.integers(low=-Env.WIDTH/2, high=(Env.HEIGHT/2)+1) 

            self.player.locate(px,py)
            self.goal.locate(gx,gy)



        self.interaction = 0
        obs = self.get_observation_position()
        self.xstart = obs[0]
        self.ystart = obs[1]
        self.progress_prev = 0
        self.action_prev = [0, 0]
        info = {} 
        
        return obs[3:], info 


    def step(self, action):
        self.interaction += 1

        # Aplicar acción
        velocity, AngleWheel = action
        AngleWheel = np.array(AngleWheel).astype(np.float32)
        self.player.move(velocity, AngleWheel)
        
        p.stepSimulation()

        # Obtener observación
        new_observation_space_position = self.get_observation_position()

        # - Calcula Recompensa 
        reward = 5 * new_observation_space_position[3]
        reward += 2 * new_observation_space_position[5] * abs(new_observation_space_position[3])
        reward += 4000 * (new_observation_space_position[5] - self.progress_prev_position)
        reward += 3 * (action[1] - self.action_prev[1])
        reward -= 0.0015 * self.interaction

        self.progress_prev_position = new_observation_space_position[5]
        self.action_prev = action
        
        if abs(new_observation_space_position[5]) > 0.95:
            reward += 10
            done = True
        else:
            done = False

        if self.interaction >= self.maxinteractions or new_observation_space_position[5] <= -0.5:
            truncated = True
            reward -= 5

        else:
            truncated = False

        new_observation_space  = new_observation_space_position

        info = {}

        return new_observation_space[3:], reward, done, truncated, info  # Devuelve el nuevo estado, la recompensa, si el episodio ha terminado y la información adicional

    def close(self):
        p.disconnect()