from scipy import stats


class Distribution:
    def __init__(self, name: str, d: stats.rv_continuous, args: dict):
        self.name = name
        self.d = d
        self.args = args


class Normal(Distribution):
    def __init__(self, loc: float = 0, scale: float = 1):
        super().__init__("Normal", stats.norm, {"loc": loc, "scale": scale})


class SkewNormal(Distribution):
    def __init__(self, a, loc: float = 0, scale: float = 1):
        super().__init__("SkewNormal", stats.skewnorm, {"a": a, "loc": loc, "scale": scale})


class Exponential(Distribution):
    def __init__(self, loc: float = 0, scale: float = 1):
        super().__init__("Exponential", stats.expon, {"loc": loc, "scale": scale})


class Beta(Distribution):
    def __init__(self, a: float = 0, b: float = 1):
        super().__init__("Beta", stats.beta, {"a": a, "b": b})


class Cauchy(Distribution):
    def __init__(self, loc: float = 0, scale: float = 1):
        super().__init__("Cauchy", stats.cauchy, {"loc": loc, "scale": scale})


class DoubleGamma(Distribution):
    def __init__(self, a: float, loc: float = 0, scale: float = 1):
        super().__init__("DoubleGamma", stats.dgamma, {"a": a, "loc": loc, "scale": scale})


class DoubleWeibull(Distribution):
    def __init__(self, c: float, loc: float = 0, scale: float = 1):
        super().__init__("DoubleWeibull", stats.dweibull, {"c": c, "loc": loc, "scale": scale})
