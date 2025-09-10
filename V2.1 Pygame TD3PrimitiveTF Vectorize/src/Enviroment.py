import pygame
import math
import numpy as np
import random

# Inicializar pygame
pygame.init()

# Configuración de la pantalla
WIDTH, HEIGHT = 800, 800

# Colores
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0 ,0 ,0)

x_zero = WIDTH/2
y_zero = HEIGHT/2

# Clase del jugador
class Player(pygame.sprite.Sprite):
    def __init__(self, x_start=0, y_start=0):
        super().__init__()

        self.x = x_start
        self.y = y_start

        x_start += x_zero
        y_start *= -1
        y_start += y_zero

        self.original_image = pygame.Surface((94, 94),pygame.SRCALPHA)  # Imagen original
        center = 47
        pygame.draw.polygon(self.original_image, GREEN, [
            (center + 28.6, center + 35.0),
            (center + 35.0, center + 0),
            (center + 28.6, center - 35.0),
            (center - 28.6, center - 35.0),
            (center - 35.0, center + 0),
            (center - 28.6, center + 35.0)
        ])  # Forma Hexagonal
        self.image = self.original_image  # Imagen que rota
        self.rect = self.image.get_rect(center=(x_start, y_start))
        self.mask = pygame.mask.from_surface(self.image) #Arregla el hitbox para que sea igual a la imagen(hexagono)
        self.angle = 0  # Ángulo en grados
        self.speed = 0  # Velocidad inicial

        #TD3
        #Observacion - Acciones - Valor Maximo - Valor Minimo        
        self.observation_space = np.array([0,0,0,0,0,0])  # Espacio de observación (PosX,PosY,Angulo,Velx,Vely,Alineacion,Errx,Erry,Delta)
        self.action_space = np.array([0,0])       # Espacio de acción de tamaño 1x2

        self.action_space_high = np.array([10, 5]) 
        self.action_space_low = np.array([-10, -5])         

    def move(self,Velocity,Vang):
        self.angle += Vang
        self.speed  = Velocity

    def update_location(self):
        x,y = self.rect.center
        return x-x_zero,-y+y_zero

    def update(self):
        # Rotar la imagen
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        # Moverse en la dirección del ángulo
        rad = math.radians(self.angle)  
        self.rect.x += self.speed * math.cos(rad+math.pi/2)
        self.rect.y -= self.speed * math.sin(rad+math.pi/2)  # Restamos porque en Pygame el eje Y crece hacia abajo

    def locate(self,x_move=0,y_move=0):
        x_move += x_zero
        y_move *= -1
        y_move += y_zero
        self.rect.centerx = x_move
        self.rect.centery = y_move
        self.angle = 0

# Clase Meta
class Goal(pygame.sprite.Sprite):
    def __init__(self, x_start=0, y_start=100):
        super().__init__()

        self.x = x_start
        self.y = y_start

        x_start += x_zero
        y_start *= -1
        y_start += y_zero

        self.image = pygame.Surface((30, 30), pygame.SRCALPHA)  # Superficie transparente
        center = 15  # Centro de la superficie

        radio = random.randint(1, 15)
        pygame.draw.circle(self.image, WHITE, (center, center), radio)

        # Definir rectángulo y máscara de colisión
        self.rect = self.image.get_rect(center=(x_start, y_start))
        self.mask = pygame.mask.from_surface(self.image)  # Hitbox real

    def update_location(self):
        x,y = self.rect.center
        return x-x_zero,-y+y_zero

    def locate(self,x_move=0,y_move=0):
        x_move += x_zero
        y_move *= -1
        y_move += y_zero
        self.rect.centerx=x_move
        self.rect.centery=y_move

# Clase obstaculo
class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x_start=0, y_start=0):
        super().__init__()

        x_start += x_zero
        y_start *= -1
        y_start += y_zero

        self.image = pygame.Surface((100, 100), pygame.SRCALPHA)  # Superficie transparente
        forma = random.choice(["rectangulo", "circulo", "pentagono", "hexagono"])
        center = 50  # Centro de la superficie

        if forma == "rectangulo":
            w, h = random.randint(10, 100), random.randint(10, 100)
            pygame.draw.rect(self.image, RED, (center - w//2, center - h//2, w, h))

        elif forma == "circulo":
            radio = random.randint(30, 60)
            pygame.draw.circle(self.image, RED, (center, center), radio)

        elif forma == "pentagono":
            puntos = []
            for i in range(5):
                angulo = math.radians(i * 72)  # 360° / 5 lados
                radio = random.randint(30, 60)
                px = center + radio * math.cos(angulo)
                py = center + radio * math.sin(angulo)
                puntos.append((px, py))
            pygame.draw.polygon(self.image, RED, puntos)

        elif forma == "hexagono":
            puntos = []
            for i in range(6):
                angulo = math.radians(i * 60)  # 360° / 6 lados
                radio = random.randint(30, 60)
                px = center + radio * math.cos(angulo)
                py = center + radio * math.sin(angulo)
                puntos.append((px, py))
            pygame.draw.polygon(self.image, RED, puntos)

        # Definir rectángulo y máscara de colisión
        self.rect = self.image.get_rect(center=(x_start, y_start))
        self.mask = pygame.mask.from_surface(self.image)  # Hitbox real
