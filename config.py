import yaml
from types import SimpleNamespace


class Config(SimpleNamespace):
    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._to_namespace(data)

    @classmethod
    def _to_namespace(cls, data):
        if isinstance(data, dict):
            return cls(**{k: cls._to_namespace(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [cls._to_namespace(v) for v in data]
        return data

    def to_dict(self):
        return self._to_dict(self)

    @classmethod
    def _to_dict(cls, obj):
        if isinstance(obj, SimpleNamespace):
            return {k: cls._to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, list):
            return [cls._to_dict(v) for v in obj]
        return obj