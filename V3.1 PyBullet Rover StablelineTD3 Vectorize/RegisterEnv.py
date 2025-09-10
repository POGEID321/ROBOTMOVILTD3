from gymnasium.envs.registration import register

register(
    id="Milimars-v0-Bullet",
    entry_point="src.EnvGym:EnviromentGymBullet",  # ruta donde se encuentra la clase
)