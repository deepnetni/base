import pandas as pd
import argparse
import json
import re


def parser():
    parser = argparse.ArgumentParser(
        description="python json2excel.py --src xx.json --out yy.xlsx"
    )
    parser.add_argument(
        "--json",
        help="json file",
        type=str,
        default="",
    )
    parser.add_argument("--out", help="out excel file path", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    with open(args.json, "r") as fp:
        data = json.load(fp)

    ctx = {}
    for k, v in data.items():
        idx = re.findall("^\d+\.?\d*", k)
        if idx != []:
            idx = int(idx[0])
            ctx.update({idx: dict(name=k, **v)})

    # print(ctx)
    df = pd.DataFrame(ctx).T
    df = df.sort_index()

    df.to_excel(args.out, index=False) if args.out != "" else None
