from gymnasium.envs.registration import register

register(
    id="Milimars-v0",
    entry_point="src.EnvGym:EnviromentGym",  # ruta donde se encuentra la clase
)