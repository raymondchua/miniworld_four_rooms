# Import the envs module so that envs register themselves

# Import the envs module so that envs register themselves
# from . import envs

# Import wrappers so it's accessible when installing with pip
#from . import wrappers

def _register_envs():
    import gym_miniworld.envs  # delay the import to avoid circular issues

_register_envs()
