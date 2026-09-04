"""One-shot converter for the legacy MuJoCo XML assets.

45 of the 89 files under envs/mujoco/assets use coordinate="global", dropped in
MuJoCo 2.3.3. Hopper and walker2d also carry unevaluated literals such as
pos="0.13/2 0 0.1" that only mujoco-py's parser tolerated.

Builds its own mujoco==2.3.3 venv, since that conflicts with the 3.10 runtime.
Output is checked in, so this runs once.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO / "envs" / "mujoco" / "assets"
DEFAULT_OUT = REPO / "comp_flow_jax" / "envs" / "assets_v3"

# Unevaluated arithmetic in body pos attributes, straight from gym's assets.
LITERAL_FIXES = [
    (re.compile(r'pos="0\.13/2 '), 'pos="0.065 '),   # hopper foot
    (re.compile(r'pos="0\.2/2 '), 'pos="0.1 '),      # walker2d foot
]

# Runs inside the mujoco==2.3.3 venv.
_WORKER = '''
import glob, os, sys, mujoco
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
ok, failed = 0, []
for f in sorted(glob.glob(os.path.join(src, "*.xml"))):
    name = os.path.basename(f)
    try:
        m = mujoco.MjModel.from_xml_path(f)
        mujoco.mj_saveLastXML(os.path.join(out, name), m)
        ok += 1
    except Exception as e:
        failed.append((name, str(e).replace("\\n", " ")[:120]))
for name, err in failed:
    print("FAIL", name, err)
print("OK", ok, "FAIL", len(failed))
sys.exit(1 if failed else 0)
'''


def _patch_literals(src: pathlib.Path, staging: pathlib.Path) -> int:
    staging.mkdir(parents=True, exist_ok=True)
    patched = 0
    for xml in sorted(src.glob("*.xml")):
        text = xml.read_text()
        original = text
        for pattern, repl in LITERAL_FIXES:
            text = pattern.sub(repl, text)
        if text != original:
            patched += 1
        (staging / xml.name).write_text(text)
    return patched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=pathlib.Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"[convert_assets] source directory not found: {args.src}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        staging = tmp / "patched"
        n_patched = _patch_literals(args.src, staging)
        n_total = len(list(staging.glob("*.xml")))
        print(f"[convert_assets] staged {n_total} assets, patched literals in {n_patched}")

        venv = tmp / "mj233"
        print("[convert_assets] building mujoco==2.3.3 venv (this conflicts with the "
              "runtime mujoco, hence the separate env)...")
        subprocess.run(["uv", "venv", "--python", "3.11", str(venv)],
                       check=True, capture_output=True)
        subprocess.run(["uv", "pip", "install", "--python", str(venv / "bin" / "python"),
                        "mujoco==2.3.3"], check=True, capture_output=True)

        worker = tmp / "worker.py"
        worker.write_text(_WORKER)
        args.out.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [str(venv / "bin" / "python"), str(worker), str(staging), str(args.out)],
            capture_output=True, text=True)
        print(proc.stdout.strip())
        if proc.returncode != 0:
            print(proc.stderr.strip(), file=sys.stderr)
            return proc.returncode

    # Verify the output actually loads under the runtime MuJoCo.
    import mujoco  # noqa: E402  (deliberately after conversion)
    bad = []
    for xml in sorted(args.out.glob("*.xml")):
        try:
            mujoco.MjModel.from_xml_path(str(xml))
        except Exception as e:  # pragma: no cover - conversion is verified in CI
            bad.append((xml.name, str(e)[:100]))
    if bad:
        for name, err in bad:
            print(f"[convert_assets] mujoco {mujoco.__version__} cannot load {name}: {err}")
        return 1
    print(f"[convert_assets] all {len(list(args.out.glob('*.xml')))} assets load "
          f"under mujoco {mujoco.__version__} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
