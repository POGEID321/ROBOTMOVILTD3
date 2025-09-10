import pygame

class GameSirController:
    def __init__(self, index=0):
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(index)
        self.joystick.init()

        print(f"Control detectado: {self.joystick.get_name()}")

        # Mapeo de botones (excepto gatillos, que los tratamos como ejes)
        self.botones = {
            0: "A",
            1: "B",
            2: "X",
            3: "Y",
            4: "LB",
            5: "RB",
            8: "BOTON_RARO_IZQ",
            9: "BOTON_RARO_DER",
            10: "PRESS_JOYSTICK_IZQ",
            11: "PRESS_JOYSTICK_DER",
            12: "CRUZ_ARRIBA",
            13: "CRUZ_ABAJO",
            14: "CRUZ_IZQ",
            15: "CRUZ_DER"
        }

    def read_state(self):
        pygame.event.pump()

        # Sticks
        LX = round(self.joystick.get_axis(0),4)
        LY = round(self.joystick.get_axis(1),4)
        RX = round(self.joystick.get_axis(2),4)
        RY = round(self.joystick.get_axis(3),4)

        # Gatillos: probamos a leerlos como ejes
        LT = 0.0
        RT = 0.0

        LT = (round(self.joystick.get_axis(4),3) + 1) / 2.0  # Normalizado 0→1
        RT = (round(self.joystick.get_axis(5),3) + 1) / 2.0  # Normalizado 0→1

        # Estado final
        state = {
            "sticks": {"LX": LX, "LY": LY, "RX": RX, "RY": RY},
            "triggers": {"LT": LT, "RT": RT},
            "buttons": {}
        }

        # Botones digitales
        for i in range(self.joystick.get_numbuttons()):
            if i in self.botones:  # ignoramos LT/RT aquí
                state["buttons"][self.botones[i]] = bool(self.joystick.get_button(i))

        return state
