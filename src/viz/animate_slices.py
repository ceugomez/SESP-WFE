# animate_slices.py
# cgf
#
# Stitch per-frame slice PNGs (from plot_xy_slices.py / plot_xz_slices.py)
# into per-variable animated GIFs. Directory-agnostic: point it at a single
# slices directory, or at a run output dir to animate xy_slices/ and
# xz_slices/ together.
#
# Usage:
#   python animate_slices.py <dir> [--vars u v w] [--duration MS]
#     <dir> = a slices dir (containing *_u.png ...), OR a run output dir
#             containing xy_slices/ and/or xz_slices/ subdirs.

import argparse
import glob
import os
import re
from PIL import Image

# frame index sits between the leading ".<step>" and the "_xy20"/"_xz" tag
STEP_RE = re.compile(r"\.(\d+)_x[yz]")


def _step_key(path):
    m = STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def animate_dir(slices_dir, variables, duration):
    """Make one GIF per (slice tag, variable) in slices_dir. Returns count made.

    A 'tag' is the plane/height label in the filename (e.g. xy20, xy50, xz), so
    a directory mixing multiple AGL heights yields one GIF per height — never an
    interleaved animation.
    """
    made = 0
    for var in variables:
        tag_re = re.compile(rf"_(x[yz]\d*)_{re.escape(var)}\.png$")
        groups = {}
        for f in glob.glob(os.path.join(slices_dir, f"*_{var}.png")):
            m = tag_re.search(os.path.basename(f))
            if m:
                groups.setdefault(m.group(1), []).append(f)
        for tag, files in sorted(groups.items()):
            files.sort(key=_step_key)
            frames = [Image.open(f) for f in files]
            out = os.path.join(slices_dir, f"{tag}_{var}_animation.gif")
            frames[0].save(out, save_all=True, append_images=frames[1:],
                           duration=duration, loop=0)
            print(f"  saved {len(frames)} frames -> {out}")
            made += 1
    return made


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="a slices dir, or a run output dir with "
                                    "xy_slices/ and/or xz_slices/ subdirs")
    parser.add_argument("--vars", nargs="+", default=["u", "v", "w"],
                        help="variables to animate (default: u v w)")
    parser.add_argument("--duration", type=int, default=100,
                        help="per-frame duration in ms (default: 100)")
    args = parser.parse_args()

    # Resolve which directories to animate: if the given dir has slice subdirs,
    # use those; otherwise treat the dir itself as a slices dir.
    subdirs = [os.path.join(args.dir, s) for s in ("xy_slices", "xz_slices")
               if os.path.isdir(os.path.join(args.dir, s))]
    targets = subdirs or [args.dir]

    total = 0
    for d in targets:
        print(f"Animating {d}")
        total += animate_dir(d, args.vars, args.duration)
    if total == 0:
        raise SystemExit(f"No slice PNGs found under {args.dir}")
    print(f"Wrote {total} GIF(s).")
