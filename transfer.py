class TransferFunctor:
    def __call__(self, omega):
        raise NotImplementedError

    def __mul__(self, other):
        a, b = self, other

        class ProductFunctor(TransferFunctor):
            def __call__(self, omega):
                return a(omega) * b(omega)

        return ProductFunctor()


class LowF(TransferFunctor):
    def __init__(self, omega0, y):
        self.omega0 = omega0
        self.y = y

    def __call__(self, omega):
        if omega < self.omega0:
            return self.y * 1.0
        # elif omega == self.omega0:
        #     return (1 + self.y) / 2.0
        else:
            return 1.0


class HighF(TransferFunctor):
    def __init__(self, omega0, y):
        self.omega0 = omega0
        self.y = y

    def __call__(self, omega):
        if omega > self.omega0:
            return self.y * 1.0
        # elif omega == self.omega0:
        #     return (1 + self.y) / 2.0
        else:
            return 1.0


class BandF(TransferFunctor):
    def __init__(self, omegaL, omegaR, y):
        self.omegaL = omegaL
        self.omegaR = omegaR
        self.y = y

    def __call__(self, omega):
        if self.omegaL < omega < self.omegaR:
            return self.y * 1.0
        # elif omega in [self.omegaL, self.omegaR]:
        #     return (1 + self.y) / 2.0
        else:
            return 1.0


class Unity(TransferFunctor):
    def __call__(self, omega):
        return 1


def BandF2(omegaL, omegaR, y):
    return HighF(omegaL, y) * HighF(omegaR, 1 / y)
