import numpy as np
from typing import Union, Dict, Optional
import torch


class REC(object):
    def __init__(self):
        self._state_dict = {}

    def state_dict(self):
        v = {k: round(v[0] / v[1], 4) for k, v in self._state_dict.items()}
        return v

    def reset(self):
        self._state_dict = {}

    def update_by_key(self, v: Union[float, torch.Tensor], key: str):
        assert key is not None
        v = v.item() if isinstance(v, torch.Tensor) else v
        value, num = self._state_dict.get(key, [0.0, 0])
        self._state_dict.update({key: [value + v, num + 1]})

    def update(self, v: Union[Dict, float, torch.Tensor], key: Optional[str] = None):
        if isinstance(v, dict):
            for k, v in v.items():
                self.update_by_key(v, k)
        else:  # v is float
            self.update_by_key(v, key)

    def __str__(self):
        ret = ""
        for k, v in self._state_dict.items():
            ret += f"key: {k}, {v}\n"
        return ret


if __name__ == "__main__":
    rec = REC()
    rec.update(10, "a")
    rec.update(15, "a")
    rec.update({"a": 10, "b": 13, "c": 11})
    print(rec)
    print(rec.state_dict())
