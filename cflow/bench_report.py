"""Render the large-scale benchmark table from runs/bench/*.json."""
import glob
import json
import os

CFGS = ["zlab", "mask_ws", "rcf_v2_k1", "mask_ws_n16", "rcf_v2_n16"]
DSETS = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]


def main():
    data = {}
    for f in glob.glob("runs/bench/*.json"):
        base = os.path.basename(f)[:-5]
        for c in sorted(CFGS, key=len, reverse=True):
            if base.startswith(c + "_"):
                ds = base[len(c) + 1:]
                data[(c, ds)] = json.load(open(f))
                break

    for metric in ("tau", "tok_per_s"):
        print(f"\n== {metric} ==")
        print(f"{'dataset':10s} " + " ".join(f"{c:>12s}" for c in CFGS))
        for ds in DSETS:
            row = [f"{ds:10s}"]
            for c in CFGS:
                r = data.get((c, ds))
                row.append(f"{r[metric]:12.3f}" if r else f"{'-':>12s}")
            print(" ".join(row))
    done = len(data)
    print(f"\n{done}/25 evals complete")


if __name__ == "__main__":
    main()
