# Import the envs module so that envs register themselves

# Import the envs module so that envs register themselves
# from . import envs

# Import wrappers so it's accessible when installing with pip
#from . import wrappers
def register():
    """Call this to register MiniWorld environments with Gym."""
    import gym_miniworld.envs
