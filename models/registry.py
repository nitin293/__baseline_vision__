MODEL_REGISTRY = {}


def register(name):
    def deco(cls):
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return deco
