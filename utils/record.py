import numpy as np
from typing import Union, Dict, Optional
import torch


class REC(object):
    def __init__(self, keep: bool = False):
        self._state_dict = {}
        self.keep = keep

    def state_dict(self):
        v = {k: round(v[0] / v[1], 4) for k, v in self._state_dict.items()}
        return v

    def reset(self):
        self._state_dict = {}

    def update_by_key(self, v: Union[float, torch.Tensor], key: str):
        assert key is not None
        v = v.item() if isinstance(v, torch.Tensor) else v
        value, num, hist = self._state_dict.get(key, [0.0, 0, []])
        if self.keep:
            hist.append(v)
        self._state_dict.update({key: [value + v, num + 1, hist]})

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


class RECDepot(object):
    def __init__(self, keep: bool = False) -> None:
        self.depots = {}
        self.keep = keep

    def update(self, k: str, v: Dict):
        rec = self.depots.get(k, REC(self.keep))
        rec.update(v)
        self.depots.update({k: rec})

    def state_dict(self):
        v = {k: rec.state_dict() for k, rec in self.depots.items()}
        return v

    def __str__(self):
        ret = ""
        for k, d in self.depots.items():
            ret += f"# Cond {k}\n{d}\n"
            # for v1, v2 in d.items():
            #     ret += f"v:{v1}\n"
        return ret


if __name__ == "__main__":
    rec = REC(True)
    rec.update(10, "a")
    rec.update(15, "a")
    rec.update({"a": 10, "b": 13, "c": 11})
    print("#", rec)
    print(rec.state_dict())

    rec = RECDepot(True)
    rec.update("a", {"1": 2, "2": 3})
    rec.update("a", {"1": 3, "2": 3})
    rec.update("c", {"1": 2, "2": 3})
    print(rec)
    print(rec.state_dict())
