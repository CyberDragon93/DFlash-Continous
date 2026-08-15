"""Summarize eval results into a comparison table."""
import glob
import json


def main():
    rows = []
    for f in sorted(glob.glob("runs/*/eval_*.json")) + sorted(glob.glob("runs/ref_zlab_*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        arm = r["draft"].split("/")[-2] if "/" in r["draft"] else r["draft"]
        rows.append({
            "arm": arm, "dataset": r["dataset"], "K": r.get("steps", 1),
            "harden": r.get("rcf_harden", False), "tau": r["tau"],
            "cycles": r["cycles"], "tok_s": r.get("tok_per_s"),
            "off1": (r.get("offset_match_rate") or [None])[0],
        })
    rows.sort(key=lambda x: (x["dataset"], x["arm"], x["K"]))
    cur = None
    for r in rows:
        if r["dataset"] != cur:
            cur = r["dataset"]
            print(f"\n== {cur} ==")
            print(f"{'arm':16s} {'K':>2s} {'tau':>7s} {'off1':>6s} {'cycles':>7s} {'tok/s':>7s}")
        h = " (harden)" if r["harden"] else ""
        print(f"{r['arm']+h:16s} {r['K']:2d} {r['tau']:7.3f} "
              f"{(r['off1'] if r['off1'] is not None else 0):6.3f} {r['cycles']:7d} {r['tok_s']:7.1f}")


if __name__ == "__main__":
    main()
