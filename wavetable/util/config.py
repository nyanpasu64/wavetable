from dataclasses import dataclass


@dataclass
class ConfigMixin:
    @classmethod
    def new(cls, **state):
        """ Redirect `Alias(key)=value` to `key=value`.
        Then call the dataclass constructor (to validate parameters). """

        for key, value in dict(state).items():
            class_var = getattr(cls, key, None)

            if class_var is Ignored:
                del state[key]

            if isinstance(class_var, Alias):
                target = class_var.key
                if target in state:
                    raise TypeError(
                        f'{type(self).__name__} received both Alias {key} and '
                        f'equivalent {target}'
                    )

                state[target] = value
                del state[key]

        return cls(**state)


@dataclass
class Alias:
    """
    @register_config
    class Foo:
        x: int
        xx = Alias('x')     # do not add a type hint
    """
    key: str


Ignored = object()
