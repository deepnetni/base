import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.compute_metrics import compute_box, to_excel
import numpy as np

if __name__ == "__main__":
    # a = np.arange(9, dtype=np.float32)
    # np.random.shuffle(a)
    # print(a)
    # b = np.percentile(a, 50)
    # print(b)
    # a = a.tolist()

    a = {
        "car": [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 2, "b": 3, "c": 1},
            {"a": 2, "b": 3},
            {"a": 2.5, "b": 3.5},
        ],
        "bus": [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 2, "b": 3, "c": 1},
            {"a": 2, "b": 3},
            {"a": 2.5, "b": 3.5},
        ],
    }
    to_excel(a, "test.xlsx")
