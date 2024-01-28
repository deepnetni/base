import configparser
import re
from typing import Dict


def read_ini(fname: str) -> Dict:
    config = configparser.ConfigParser()
    config.read(fname, encoding="utf-8")
    contents = {}

    sections = config.sections()
    for sec in sections:
        contents[sec] = {}
        options = config.options(sec)
        for opt in options:
            value = config.get(sec, opt)
            if re.search(".*_int$", opt) is not None:
                contents[sec][opt.rstrip("_int")] = int(value)
            elif re.search(".*_bool$", opt) is not None:
                contents[sec][opt.rstrip("_bool")] = bool(value.lower() == "true")
            else:
                contents[sec][opt] = value
    return contents


if __name__ == "__main__":
    cfg = read_ini("../config.ini")
    print(cfg)
