import math
import time
import numpy as np
import tensorflow as tf
import pygame
import src.TD3 as Agente
import src.Enviroment as Env 
import threading
import time

# Crear Enviroment
player = Env.Player()
goal = Env.Goal()
Scene = Env.pygame.sprite.Group(player,goal) #, *obstacles)

Pgoal = [[0, 100], [100, 100], [100, 0], [100, -100], [0, -100], [-100, -100], [-100, 0], [-100, 100]]
Pplayer = [player.x,player.y]

def Sim():

    # while 1:

        for event in Env.pygame.event.get():
            if event.type == Env.pygame.QUIT:
                running = False 
        
        # Actualizar y dibujar
        Env.screen.fill(Env.BLACK)
        Scene.update()
        Scene.draw(Env.screen)
        Env.pygame.display.flip()
        Env.clock.tick(30)

def StartSim():

    Simulation = threading.Thread(target=Sim)
    Simulation.start()

def Observation():

    xplayer, yplayer = player.update_location()
    xgoal, ygoal = goal.update_location()
    angle=math.radians(player.angle)
    Alineacion = math.cos(math.atan2(ygoal-yplayer,xgoal-xplayer)-(player.angle+90)*math.pi/180)
    ErrX = xgoal - xplayer
    ErrY = ygoal - yplayer
    Delta_Dis = np.linalg.norm([xgoal - xplayer, ygoal - yplayer])


    return np.array([round(xplayer/100,4),round(yplayer/100,4),
                     round(angle/(math.pi/2),4),round(Alineacion,4),
                     round(ErrX/100,4),round(ErrY/100,4),
                     round(Delta_Dis/100,4)])

def Execute(action,con_initial): 

    #- Ejecuta Velocidad 
    Vlineal=tf.cast(action[0],tf.float32)
    Vang=tf.cast(action[1],tf.float32)
    player.move(Vlineal,Vang)
    # time.sleep(0.1)
    # - Lee Datos 
    new_observation_space = Observation()
    
    xplayer, yplayer = new_observation_space[0], new_observation_space[1]
    xgoal, ygoal = goal.update_location()
    xgoal, ygoal = xgoal/100, ygoal/100

    Alineacion = math.cos(math.atan2(ygoal-yplayer,xgoal-xplayer)-(player.angle+90)*math.pi/180) #Producto entre 2 vectores Partefrontal y trasera del rover al punto objetivo 
    # - Calcula Recompensa 
    xstart=con_initial[0]
    ystart=con_initial[1]
    Mov = (0.1*action[0]**2)+(0.1*action[1]**2)
    #Calculo de progreso de recorrido
    initial_distance = np.linalg.norm([xgoal - xstart, ygoal - ystart])
    actual_distance = np.linalg.norm([xgoal - xplayer, ygoal - yplayer])
    delta_distance = initial_distance - actual_distance

    reward = 0.01*Mov + 3*delta_distance/initial_distance  + 2*Alineacion 
    
    print("V:",round(action[0].numpy().item(),4),
        " ","W:",round(action[1].numpy().item(),4),
        " ","x:"," ",round(xplayer,4),
        " ","y:"," ",round(yplayer,4),
        " ","Delta Dis:"," ",round(actual_distance,4),
        " ","Delta/Inicial:"," ",round(delta_distance/initial_distance,4),
        " ","Angulo:"," ",round(player.angle.numpy().item()+90,4),
        " ","Alineacion:"," ",round(Alineacion,4),
        " ","Reward:",round(reward.numpy().item(),4))
    
    print(new_observation_space)

    done = False 
    # if Env.pygame.sprite.collide_mask(player, goal):
    #     done = True    
    if abs(actual_distance) <= 0.1:
        reward += 1
        done = True


    return new_observation_space, reward, done, {}  # Devuelve el nuevo estado, la recompensa, si el episodio ha terminado y la información adicional

Control_Automatico = Agente.Agent(alpha=0.001, beta=0.001, input_dims=player.observation_space.shape,
                  tau=0.005, env=player, batch_size=64, noise=0.1)

n_Episodes = 500
Meta = 0
Pos = 0
cont_Done = 0

# StartSim()

for i in range(n_Episodes):

    done = False 
    score = 0
    cont = 0
    obs_start = Observation()

    while not done and cont < 300:
        
        for event in Env.pygame.event.get():
            if event.type == Env.pygame.QUIT:
                running = False 

        cont+=1
        print("Ep:",i,",Int:",cont,"Task Complete:",Meta)
        observation = Observation()

        #Control Automatico
        action = Control_Automatico.choose_action(observation)
        
        #Control Manual
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
        
        new_observation, reward, done, _ = Execute(action,obs_start)
        reward = tf.clip_by_value(reward, -10.0, 10.0)
        new_observation = new_observation
        Control_Automatico.remember(observation, action, reward, new_observation, done)
        Control_Automatico.learn()
        score += reward.numpy().item()
        observation = new_observation

        Sim()

        if done == True:
            Meta+=1

            if cont_Done < 10:
                cont_Done += 1
            else:
                cont_Done = 0
                if Pos<8:
                    Pos += 1
                else:
                    Pos = 0

            # player.locate(Env.random.randint(-100, 100),Env.random.randint(-100, 100))
            # goal.locate(Env.random.choice(list(range(-400, -100)) + list(range(100, 400))),Env.random.choice(list(range(-400, -100)) + list(range(100, 400))))

            # Pplayer[0] = Env.random.randint(-100, 100)
            # Pplayer[1] = Env.random.randint(-100, 100)
            # Pgoal[0] = Env.random.choice(list(range(-200, -100)) + list(range(100, 200)))
            # Pgoal[1] = Env.random.choice(list(range(-200, -100)) + list(range(100, 200)))
            print("EXITO")
            break
    
    #Control_Automatico.Ajuste_Ruido()

    player.locate(Pplayer[0],Pplayer[1])
    goal.locate(Pgoal[Pos][0],Pgoal[Pos][1])

    #INTERRUPCION POR VECES LOGRADO
    if Meta == 100:
        break
    print("Ep", i, 'score %.2f' % score)

    # Actualizar y dibujar
    Env.screen.fill(Env.BLACK)
    Scene.update()
    Scene.draw(Env.screen)
    Env.pygame.display.flip()
    Env.clock.tick(30)

Env.pygame.quit()
