import importlib

# Lazy import: each algorithm is imported only when it is actually used,
# so unfinished PPO/SAC code does not break imitation learning (and vice versa).
_ALGO_MODULES = {
    'imitation': 'algorithm.imitation',
    'ppo': 'algorithm.ppo',
    'sac': 'algorithm.sac',
    }


class _LazyAlgoDict(dict):
    def __getitem__(self, key):
        if key not in _ALGO_MODULES:
            raise KeyError(key)
        return importlib.import_module(_ALGO_MODULES[key]).Agent

    def __contains__(self, key):
        return key in _ALGO_MODULES

    def keys(self):
        return _ALGO_MODULES.keys()


algo_dict = _LazyAlgoDict()
