import numpy as np
from td3_tf2 import Agent
import sim
import math
import tensorflow as tf

#COMUNICACION SIMULACION --COPPELIA
sim.simxFinish(-1) 
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) 
if clientID != -1: print("conectado ")
else: print("no se pudo conectar")

Meta = 0   # Meta Lograda
cont=0     # # de Interaccion

# Ubicacion inicial
IX= 0.0     #Inicial X
IY= 0.0     #Inicial Y


# Ubicacion deseada
FX= 1       #Final X
FY= 1       #Final Y

# Define un entorno ficticio para propósitos de ejemplo
class DummyEnvironment: 
    def __init__(self):

        #COMUNICACION SIMULACION --COPPELIA
        sim.simxFinish(-1) 
        self.clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) 
        if self.clientID != -1: print("conectado ")
        else: print("no se pudo conectar")

        #--COPPELIA || DATOS A MODIFICAR Y LEER 
        returnCode,Carro=sim.simxGetObjectHandle(self.clientID,'Ubicacion2',sim.simx_opmode_blocking)
        returnCode,position=sim.simxGetObjectPosition(self.clientID,Carro,-1,sim.simx_opmode_blocking)
        returnCode,eulerAngles=sim.simxGetObjectOrientation(self.clientID,Carro,-1,sim.simx_opmode_blocking)
        returnCode,linearVelocity,angularVelocity=sim.simxGetObjectVelocity(self.clientID,Carro,sim.simx_opmode_blocking)

        #Velocidad del chasis
        velocidadch = math.sqrt(linearVelocity[0]**2+linearVelocity[1]**2)

        #Error de posicion
        ErrX=FX-position[0]
        ErrY=FY-position[1]

        #Observacion - Acciones - Valor Maximo - Valor Minimo        
        self.observation_space = np.array([position[0],position[1],eulerAngles[2],linearVelocity[0],linearVelocity[1],angularVelocity[2],velocidadch,ErrX,ErrY])  # Espacio de observación de tamaño 2
        self.action_space = np.array([0,0])      # Espacio de acción de tamaño 1x2

        self.action_space_high = np.array([10.0,10.0]) 
        self.action_space_low = np.array([-5.0,-5.0]) 


    def reset(self):
        returnCode,Carro=sim.simxGetObjectHandle(self.clientID,'Robot_Solo',sim.simx_opmode_blocking)
        returnCode=sim.simxSetObjectPosition(clientID,Carro,-1,[IX ,IY ,0.07],sim.simx_opmode_blocking)
        returnCode,Carro=sim.simxGetObjectHandle(self.clientID,'Robot_Solo',sim.simx_opmode_blocking)
        returnCode=sim.simxSetObjectOrientation(clientID,Carro,-1,[0.0 ,0.0 ,-math.pi],sim.simx_opmode_blocking)

        return self.observation_space               # Estado inicial aleatorio de tamaño 2

    def step(self, action):

        velocidad_derecha=tf.cast(action[0],tf.float32)
        velocidad_izquierda=tf.cast(action[1],tf.float32)

        #--COPPELIA || ENVIO DE DATOS
        returnCode,RuedaDer=sim.simxGetObjectHandle(clientID,'Motor_der2',sim.simx_opmode_oneshot)
        returnCode=sim.simxSetJointTargetVelocity(clientID,RuedaDer,velocidad_derecha,sim.simx_opmode_blocking)
        returnCode,RuedaIzq=sim.simxGetObjectHandle(clientID,'Motor_izq2',sim.simx_opmode_oneshot)
        returnCode=sim.simxSetJointTargetVelocity(clientID,RuedaIzq,velocidad_izquierda,sim.simx_opmode_blocking)       

        #--COPPELIA || LECTURA DE DATOS
        returnCode,Carro=sim.simxGetObjectHandle(self.clientID,'Ubicacion2',sim.simx_opmode_blocking)
        returnCode,position=sim.simxGetObjectPosition(self.clientID,Carro,-1,sim.simx_opmode_blocking)
        returnCode,eulerAngles=sim.simxGetObjectOrientation(self.clientID,Carro,-1,sim.simx_opmode_blocking)
        returnCode,linearVelocity,angularVelocity=sim.simxGetObjectVelocity(self.clientID,Carro,sim.simx_opmode_blocking)

        #Velocidad del chasis
        velocidadch = math.sqrt(linearVelocity[0]**2+linearVelocity[1]**2)


        #Error de posicion
        ErrX=FX-position[0]
        ErrY=FY-position[1]

        #Observacion - Acciones - Valor Maximo - Valor Minimo   
        new_observation_space = np.array([position[0],position[1],eulerAngles[2],linearVelocity[0],linearVelocity[1],angularVelocity[2],velocidadch,ErrX,ErrY]) 
    
#####################################RECOMPENSA#####################################
        #MOMIVIENTO DE LLANTAS
        Mov_Rueda = (0.1*action[0]**2)+(0.1*action[1]**2)

        #DELTA DE ANGULO ENTRE ANGULO DESEADO Y ANGULO DEL CHASIS
        Angle = math.cos(math.atan2(FY-position[1],FX-position[0])-eulerAngles[2])

        #PROGRESO DE DESPLAZAMIENTO
            #Define el progreso del desplazamiento del auto en funcion del punto Final y la Inicial 
            #DELTA DE DISTANCIA ENTRE PUNTO DESEADO Y ACTUAL

        DisX = FX-IX
        DisY = FY-IY

        auxX = DisX/100
        auxY = DisY/100

        if  position[0]>IX and position[0]<FX:
            ProgresoX = position[0]*auxX
        else:
            ProgresoX = -1*abs(position[0]*auxX)

        if  position[1]>IY and position[1]<FY:
            ProgresoY = position[1]*auxY
        else:
            ProgresoY = -1*abs(position[1]*auxY)

        #ESTABILIDAD DE MOVIMIENTO
        #margen de velocidad igual

        # Menor_VIzq=action[1]-0.3
        # Mayor_VIzq=action[1]+0.3


        # if action[0]<Mayor_VIzq and Menor_VIzq<action[0] :     
        #     Coordinacion = action[1]*2
        # else:
        #     Coordinacion = 0

        reward = Mov_Rueda*0.0001 + Angle + 100*ProgresoX + 100*ProgresoY

        print("VDer:",round(action[0].numpy().item(),4),
          " ","Vizq:",round(action[1].numpy().item(),4),
          " ","Alineacion:"," ",round(Angle,4),
          " ","Progreso en X:"," ",round(100*ProgresoX,4),
          " ","Progreso en Y:"," ",round(100*ProgresoY,4),      
          " ","Reward:",round(reward.numpy().item(),4))

        DeltaX=FX-position[0]
        DeltaY=FY-position[1]

        if DeltaX<0.05 and DeltaY<0.05:
            done = True 
        else:
            done = False                                # El episodio no termina hasta que se indique

        return new_observation_space, reward, done, {}  # Devuelve el nuevo estado, la recompensa, si el episodio ha terminado y la información adicional

if __name__ == "__main__":
    env = DummyEnvironment()
    agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape,
                  tau=0.005, env=env, batch_size=64, noise=0.5)


    agent.load_models()
    n_Episodes = 13

    for i in range(n_Episodes):
        observation = env.reset()
        done = False 
        score = 0
        cont = 0

        while not done and cont < 100:

            cont+=1
            print("Ep:",i,",Int:",cont,"Task Complete:",Meta)
            # Ajustar la forma del estado
            observation = observation
            action = agent.choose_action(observation)
            new_observation, reward, done, _ = env.step(action)
            new_observation = new_observation
            agent.remember(observation, action, reward, new_observation, done)
            agent.learn()
            score +=tf.convert_to_tensor([reward],dtype=tf.float32)
            observation = new_observation
            if done == True:
                Meta+=1
                print("EXITO")
        #INTERRUPCION POR 5 VECES LOGRADO
        if Meta == 20:
            break
        print("Ep", i, 'score %.2f' % score)
    
        agent.Ajuste_Ruido()

    agent.save_models()


       