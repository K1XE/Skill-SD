def get_environment(env_type):
    if env_type != 'AlfredTWEnv':
        raise ValueError(f"Unsupported environment: {env_type}")
    from .alfred_tw_env import AlfredTWEnv
    return AlfredTWEnv
