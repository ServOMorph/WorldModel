SCENARIOS = {
    "smoke": {"seed": 0, "actions": ["noop", "left", "right", "up", "down"] * 2},
    "short": {"seed": 1, "actions": ["right"] * 30 + ["up"] * 30 + ["left"] * 30 + ["down"] * 30},
    "long": {"seed": 2, "actions": (["right", "up", "left", "down"] * 15) * 10},
}


def get(name):
    if name not in SCENARIOS:
        raise KeyError(f"scenario inconnu: {name}")
    return SCENARIOS[name]
