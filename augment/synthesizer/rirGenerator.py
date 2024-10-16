import glob
import logging
import os
import random
import pickle

import numpy as np
import rir_generator as rir
import scipy.signal as sps
import yaml
from matplotlib import pyplot as plt
from numba.core.datamodel import register

logger = logging.getLogger(__name__)


class RIRGenerator:
    def __init__(self, **kwargs) -> None:
        self.conf_rooms = {}
        self.rt60 = kwargs["rt60"]
        self.fs = kwargs["fs"]
        self.samples = kwargs["samples"]
        self.conf_mics = kwargs["array"]
        self.conf_spkr = kwargs["spker"]
        for rtype, conf in kwargs.get("rooms", {}).items():
            if conf["prop"] <= 0:
                continue

            self.conf_rooms[rtype] = conf

        props = np.array([x["prop"] for _, x in self.conf_rooms.items()], dtype="float")
        props /= sum(props)

        self.props = props
        self.types = list(self.conf_rooms.keys())

    @classmethod
    def from_yaml(cls, conf):
        with open(conf) as f:
            # conf_dict = yaml.load(f, Loader=yaml.FullLoader)
            conf_dict = yaml.load(f, Loader=yaml.CLoader)

        assert "rir_configs" in conf_dict
        return cls(**conf_dict["rir_configs"])

    def sample(self, norm=True):
        choice = np.random.choice(self.types, p=self.props)
        conf = self.conf_rooms[choice]

        # * step1. sample room size
        x, y = np.random.uniform(*conf["w"], size=2).round(4)
        (z,) = np.random.uniform(*conf["h"], size=1).round(4)

        rt60 = round(np.random.uniform(*self.rt60), 4)

        # * step2. sample mic array position
        nx, ny = np.random.uniform(*self.conf_mics["displacement"], size=2).round(4)
        # return [v] without the quote `,`
        (nz,) = np.random.uniform(*self.conf_mics["h"], size=1).round(4)
        # print(self.conf_mics)

        if self.conf_mics["put"] == "center":
            center_x = x // 2 + nx
            center_y = y // 2 + ny
            center_z = nz
        else:
            raise RuntimeError(f"{self.conf_mics['put']} not supported.")

        if self.conf_mics["type"] == "radius":
            r = self.conf_mics["radius"]
            st_angle = np.random.uniform(*self.conf_mics["angle"], size=1).round(4)
            st_angle = (st_angle * np.pi) / 180
            reso = np.arange(0, 2 * np.pi, 2 * np.pi / self.conf_mics["mics"])
            angles = st_angle + reso

            mic_array = [
                [r * np.cos(ang) + center_x, r * np.sin(ang) + center_y, center_z] for ang in angles
            ]
            # angles = np.arange(0, 2 * np.pi, 2 * np.pi / 100)
            # sx, sy = center_x + r * np.cos(angles), center_y + r * np.sin(angles)
            # plt.plot(sx, sy)
            # plt.scatter([x[0] for x in mic_array], [x[1] for x in mic_array], c="red")
            # plt.savefig("a.svg")

        else:
            raise RuntimeError(f"{self.conf_mics['type']} not supported.")

        # * step3. sample source array position
        while True:
            # float
            sr = np.random.uniform(*self.conf_spkr["dist_to_mics"])
            sa = np.random.uniform(0, 2 * np.pi)
            sx, sy = (np.cos(sa) * sr).round(4), (np.sin(sa) * sr).round(4)

            sx += center_x
            sy += center_y
            sz = center_z

            if (
                sx <= x - self.conf_spkr["dist_to_wall"]
                and sx >= self.conf_spkr["dist_to_wall"]
                and sy <= y - self.conf_spkr["dist_to_wall"]
                and sy >= self.conf_spkr["dist_to_wall"]
                and sr >= self.conf_mics["radius"]
            ):
                spk_pos = [sx, sy, sz]
                break

        # * step4. generate the rir samples
        h = rir.generate(
            c=340,  # sound velocity
            fs=self.fs,
            L=[x, y, z],  # room dimensions [x,y,z](m)
            r=mic_array,  # receiver position [x,y,z](m)
            s=spk_pos,  # source position
            reverberation_time=rt60,
            nsample=self.samples,  # number of output samples
        )  # (samples, mics)

        h = h.T  # mics, samples
        h = h / np.linalg.norm(h, axis=-1, keepdims=True) if norm else h

        # normalize the response value
        meta = {
            "room_size": [x, y, z],
            "mic_array": mic_array,
            "spk_pos": spk_pos,
            "rt60": rt60,
            "h": h,
            "norm": norm,
        }

        return meta


class RIRDict:
    def __init__(self, config):
        datasets = dict()
        weights = dict()

        for dataset_name, data_config in config.items():
            if "weight" in data_config and data_config["weight"] <= 0:
                continue

            if dataset_name in datasets:
                raise ValueError(f"Duplicate dataset '{dataset_name}'")

            files = list(glob.glob(os.path.join(data_config["dir"], "**/*.pkl"), recursive=True))
            if len(files) > 0:
                datasets[dataset_name] = files
                weights[dataset_name] = data_config["weight"]
            else:
                # raise RuntimeWarning(f"Empty dataset, ignoring: {dataset_name}")
                logger.info(f"Empty dataset, ignoring: {dataset_name}")

        # if len(weights) == 0:
        #     raise ValueError("No datasets are defined.")

        names = list(weights.keys())
        props = np.array(list(weights.values()), dtype="float")
        props /= sum(props)
        self.names = names  # name list
        self.props = props  # weights list
        self.datasets = datasets  # rir list

        # self.rng = np.random.default_rng(random_seed)

    def sample(self):
        name = np.random.choice(self.names, p=self.props)
        rir_list = self.datasets[name]
        # sample one rir from the dataset
        rir_p = random.sample(rir_list, 1)[0]

        try:
            # rirs_meta = np.load(str(rir_meta_path), allow_pickle=True)
            with open(rir_p, "rb") as f:
                rirs_meta = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {rir_p} from rir dataset")
            raise e

        return rirs_meta["h"], rirs_meta
