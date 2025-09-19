import numpy as np
import threading
import time
import cv2
from scipy.ndimage import distance_transform_edt

class OccupancyGrid2D:
    def __init__(self, width, height, resolution, decay=0.999, name="Grid", fps=10):
        """
        width, height : tamaño del mapa en metros
        resolution    : tamaño de celda en metros
        decay         : factor de decaimiento por ciclo (0-1)
        """
        self.res = resolution
        self.w = int(width / resolution)
        self.h = int(height / resolution)
        self.grid = np.zeros((self.h, self.w), dtype=float)  # log-odds o score
        self.costmap = np.zeros((self.h, self.w), dtype=np.uint8)
        self.decay = decay
        
        self.name = name
        self.fps = fps
        self._running = False
        self._thread = None

    def world_to_map(self, x, y):
        """Convierte coords reales a índice de celda."""
        mx = int(x / self.res + self.w // 2)
        my = int(y / self.res + self.h // 2)
        return mx, my

    def bresenham_line(self, x0, y0, x1, y1):
        """Algoritmo de Bresenham para obtener celdas entre dos puntos."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def update_ray(self, robot_pos, hit_point):
        """
        robot_pos: (x,y) en metros
        hit_point: (x,y) en metros (impacto del rayo)
        """
        x0, y0 = self.world_to_map(*robot_pos)
        x1, y1 = self.world_to_map(*hit_point)

        cells = self.bresenham_line(x0, y0, x1, y1)

        # todas menos la última → libres
        for (cx, cy) in cells[:-1]:
            self.grid[cy, cx] = max(self.grid[cy, cx] - 1, -5)  # libre

        # última celda → ocupada
        cx, cy = cells[-1]
        self.grid[cy, cx] = min(self.grid[cy, cx] + 3, 10)  # ocupado

    def decay_step(self):
        """Aplica decaimiento global (limpieza progresiva)."""
        self.grid *= self.decay

    def get_Grid(self):
        """Devuelve mapa binario: -1=desconocido, 0=libre, 1=ocupado."""
        OccupancyGrid = np.zeros_like(self.grid, dtype=int)
        OccupancyGrid[self.grid > 1] = 1
        OccupancyGrid[self.grid < -1] = 0
        OccupancyGrid[np.logical_and(self.grid >= -1, self.grid <= 1)] = -1
        return OccupancyGrid
    
    def reset_Grid(self):
        self.grid = np.zeros((self.h, self.w), dtype=float)

    def _updater(self):
        delay = 1.0 / self.fps
        while self._running:

            frame = cv2.normalize(self.grid, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)

            # Convertir a BGR si es 2D
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            cv2.imshow(self.name, frame)

            # cv2.waitKey necesita estar en el mismo hilo de imshow
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                self.stop()
                break

            time.sleep(delay)

        cv2.destroyWindow(self.name)
        
    def get_Costmap(self):
        grid = self.get_Grid()
        dist = distance_transform_edt(1 - grid)  
        inflation_radius = 5   # celdas
        self.costmap = np.clip((inflation_radius - dist), 0, inflation_radius)
        self.costmap = (self.costmap / inflation_radius) * 100
        
        return self.costmap   

    def run(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._updater, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
            
            

