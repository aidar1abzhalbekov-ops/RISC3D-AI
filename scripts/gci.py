"""
Grid Convergence Index (Roache) from three mesh levels with constant refinement ratio r.

Inputs: f1 (finest), f2 (medium), f3 (coarsest) scalar results, and r.
Outputs: observed order p, extrapolated value, GCI_fine (fraction), GCI_medium.

Standard formulas (Roache 1994, Celik et al. 2008):
    p      = ln((f3 - f2) / (f2 - f1)) / ln(r)           (requires f3-f2 and f2-f1 same sign)
    f_ext  = f1 + (f1 - f2) / (r**p - 1)
    e21    = |(f1 - f2) / f1|
    GCI12  = 1.25 * e21 / (r**p - 1)                     (Fs = 1.25 for 3-level studies)
"""

import math
import sys


def observed_order(f1: float, f2: float, f3: float, r: float) -> float:
    d32 = f3 - f2
    d21 = f2 - f1
    if d32 == 0.0 or d21 == 0.0:
        raise ValueError("zero difference between levels: order is undefined")
    if (d32 > 0) != (d21 > 0):
        raise ValueError("levels are not monotonically converging (sign change): no observed order")
    ratio = d32 / d21
    return math.log(abs(ratio)) / math.log(r)


def gci(f1: float, f2: float, f3: float, r: float) -> dict:
    p = observed_order(f1, f2, f3, r)
    rp = r ** p
    f_ext = f1 + (f1 - f2) / (rp - 1.0)
    e21 = abs((f1 - f2) / f1)
    gci_fine = 1.25 * e21 / (rp - 1.0)
    e32 = abs((f2 - f3) / f2)
    gci_medium = 1.25 * e32 / (rp - 1.0)
    return {
        "p": p,
        "f_ext": f_ext,
        "gci_fine": gci_fine,
        "gci_medium": gci_medium,
        "asymptotic_ratio": gci_medium / (r ** p * gci_fine) if gci_fine else float("nan"),
    }


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: gci.py f_fine f_medium f_coarse r", file=sys.stderr)
        sys.exit(2)
    f1, f2, f3, r = (float(a) for a in sys.argv[1:5])
    out = gci(f1, f2, f3, r)
    print(f"observed order p      = {out['p']:.4f}")
    print(f"extrapolated value    = {out['f_ext']:.6f}")
    print(f"GCI fine   (fraction) = {out['gci_fine']:.4%}")
    print(f"GCI medium (fraction) = {out['gci_medium']:.4%}")
    print(f"asymptotic ratio      = {out['asymptotic_ratio']:.4f}  (should be ~1.0)")
