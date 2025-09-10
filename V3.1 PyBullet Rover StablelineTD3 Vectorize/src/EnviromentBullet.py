import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
import random

WIDTH, HEIGHT = 8, 8

class Player():
    def __init__(self,x_start=0, y_start=0):
        super().__init__()

        self.x = x_start
        self.y = y_start
        if p.isConnected():

            self.plane = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.05])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=self.plane, basePosition=[0, 0, -0.05])

            # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #Plano
            p.setGravity(0, 0, -9.81) # Establecer gravedad
            # p.setTimeStep(1.0 / 240.0) 

            # self.plane = p.loadURDF("plane.urdf")  # Cargar plano base
            p.changeDynamics(self.plane, -1, lateralFriction=5.0)

            urdf_path = os.path.join(os.getcwd(),
                            "./ROVER 2025 MOVIL URDF/urdf/ROVER 2025 MOVIL URDF.urdf") # Ruta absoluta o relativa a tu URDF personalizado
            self.Robot = p.loadURDF( fileName=urdf_path, basePosition=[0, 0, 0.45], 
                                    useFixedBase=False, 
                                    baseOrientation=p.getQuaternionFromEuler([math.radians(90), 0, math.radians(-90)])) # Cargar tu robot
            
            
            p.changeDynamics(self.Robot, 2, restitution=0.0, lateralFriction=1.2)
            p.changeDynamics(self.Robot, 4, restitution=0.0, lateralFriction=1.2)
            p.changeDynamics(self.Robot, 6, restitution=0.0, lateralFriction=1.2)
            p.changeDynamics(self.Robot, 9, restitution=0.0, lateralFriction=1.2)
            p.changeDynamics(self.Robot, 13, restitution=0.0, lateralFriction=1.2)
            p.changeDynamics(self.Robot, 14, restitution=0.0, lateralFriction=1.2)

        self.speed = 0
        self.angwheel = 0
        self.angle = 0

        #TD3
        #Observacion - Acciones - Valor Maximo - Valor Minimo        
        self.observation_space = np.array([0,0,0,0])  # Espacio de observación (PosX,PosY,Angulo,Velx,Vely,Alineacion,Errx,Erry,Delta)
        self.action_space = np.array([0,0])       # Espacio de acción de tamaño 1x2

        self.action_space_high = np.array([1, 1]) 
        self.action_space_low = np.array([-1, -1])          

    def move(self,Velocity,Ang):
        
        self.speed = Velocity
        self.angwheel = Ang

        Velocity *= 2.5
        AngW = 2.1 * Ang
        L = 0.35
        W = 0.904
        Dr = 0.572
        Rw = 0.1
        Ang = 10 * AngW * math.pi/180.0

        zR = math.tanh(L*math.tan(Ang) /(L - 0.5*Dr*math.tan(Ang)))
        zL = math.tanh(L*math.tan(Ang) /(L + 0.5*Dr*math.tan(Ang)))

        if Ang != 0:

            R = L/math.tan(Ang)

            Rw = 0.1
            BoxRed =10
            
            W_fl = (Velocity * math.sqrt( L**2 + (R-(Dr/2))**2 ))/ Rw
            W_fr = (Velocity * math.sqrt( L**2 + (R+(Dr/2))**2 ))/ Rw

            if R < 0:
                W_ml = (Velocity * (R-(W/2)))/ Rw * -1
                W_mr = (Velocity * (R+(W/2)))/ Rw * -1
            else:
                W_ml = (Velocity * (R+(W/2)))/ Rw
                W_mr = (Velocity * (R-(W/2)))/ Rw

            W_bl = (Velocity * math.sqrt( L**2 + (R-(Dr/2))**2 ))/ Rw
            W_br = (Velocity * math.sqrt( L**2 + (R+(Dr/2))**2 ))/ Rw
        
        else:
            Wt = Velocity/ Rw
            W_fl, W_fr, W_ml, W_mr, W_bl, W_br = [Wt] * 6

        # print(round(float(W_fl), 4),round(float(W_fr), 4),
        #       round(float(W_ml), 4),round(float(W_mr), 4),
        #       round(float(W_bl), 4),round(float(W_br), 4),
        #       math.degrees(round(float(zL), 4)),math.degrees(round(float(zR), 4)))
 
        # # Estructura Suspension Rocker - Bogie:   0:Rocker_L   1:Bogie_L   7:Rocker_R   8:Bogie_R 
        p.setJointMotorControl2(self.Robot, 0, p.POSITION_CONTROL, targetPosition= 0, force=1000)
        p.setJointMotorControl2(self.Robot, 1, p.POSITION_CONTROL, targetPosition= 0, force=1000)
        p.setJointMotorControl2(self.Robot, 7, p.POSITION_CONTROL, targetPosition= 0, force=1000)
        p.setJointMotorControl2(self.Robot, 8, p.POSITION_CONTROL, targetPosition= 0, force=1000)

        # # Direccion:   3:Trasera_L   10:Trasera_R   5:Delantera_L   12 :Delantera_R 
        p.setJointMotorControl2(self.Robot, 3, p.POSITION_CONTROL, targetPosition= -zL, force= 10) 
        p.setJointMotorControl2(self.Robot, 10, p.POSITION_CONTROL, targetPosition= -zR, force= 10)
        p.setJointMotorControl2(self.Robot, 5, p.POSITION_CONTROL, targetPosition= zL, force= 10)
        p.setJointMotorControl2(self.Robot, 12, p.POSITION_CONTROL, targetPosition= zR, force= 10)

        # # Velocidad Wheels:   6:FL   13:FR   2:ML   9:MR   4:TL   11:TR
        p.setJointMotorControl2(self.Robot, 6, p.VELOCITY_CONTROL, targetVelocity= W_fl, force= 15) 
        p.setJointMotorControl2(self.Robot, 13, p.VELOCITY_CONTROL, targetVelocity= W_fr, force= 15) 
        p.setJointMotorControl2(self.Robot, 2, p.VELOCITY_CONTROL, targetVelocity= W_ml, force= 15) 
        p.setJointMotorControl2(self.Robot, 9, p.VELOCITY_CONTROL, targetVelocity= W_mr, force= 15) 
        p.setJointMotorControl2(self.Robot, 4, p.VELOCITY_CONTROL, targetVelocity= W_bl, force= 15) 
        p.setJointMotorControl2(self.Robot, 11, p.VELOCITY_CONTROL, targetVelocity= W_br, force= 10) 

    def update_location(self):
        position, orientation = p.getBasePositionAndOrientation(self.Robot)
        self.angle = p.getEulerFromQuaternion(orientation)[2]
        return position[0], position[1]

    def locate(self,x_move=0,y_move=0):
        p.resetBasePositionAndOrientation(  self.Robot,
                                            posObj=[x_move, y_move, 0.45],  # nueva posición
                                            ornObj=p.getQuaternionFromEuler([math.radians(90), 0, math.radians(-90)]))

class Goal():
    def __init__(self, x_start=0, y_start=100):
        super().__init__()

        self.x = x_start
        self.y = y_start
        
        height=1.0
        if p.isConnected():
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, rgbaColor=[1, 0, 0, 1])
            self.Goal = p.createMultiBody(baseVisualShapeIndex=visual_shape,
                                basePosition=[x_start, y_start, height/2])

    def update_location(self):
        position, _ = p.getBasePositionAndOrientation(self.Goal)
        return position[0], position[1]
    
    def locate(self,x_move=0,y_move=0):
        p.resetBasePositionAndOrientation(  self.Goal,
                                            posObj=[x_move, y_move, 0.45],
                                            ornObj=p.getQuaternionFromEuler([0, 0, 0]))
        
class Obstacle():
    def __init__(self, x_start=0, y_start=0):
        super().__init__()

        forma = random.choice(["rectangulo", "cilindro"])

        if forma == "rectangulo":
            half_extents = [random.uniform(0.01, 1), random.uniform(0.01, 1), 0.5]     #width, lenght, height 
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            rotation = random.uniform(0, 1.57)
            p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=collision_shape,
                              basePosition=[x_start, y_start, half_extents[2]],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, math.radians(rotation)])
                              )

        elif forma == "cilindro":
            radio = random.uniform(0.01, 1)
            height = random.uniform(0.01, 1)
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radio, height=height)
            p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=collision_shape,
                              basePosition=[x_start, y_start, height/2])
