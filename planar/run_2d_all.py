"""Re-run all five 2D methods on eth (now with valid, long scenes) and hotel (new),
via runner_2d.py. Sequential subprocess calls so each writes its traj/metric files.
"""
import subprocess, os, time

PY = "/opt/anaconda3/envs/cp/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["eth", "hotel"]
CONTROLLERS = ["cc", "acp-mpc", "ecp-mpc", "fcp-hard-adaptive", "fcp-soft-adaptive"]

for ds in DATASETS:
    for c in CONTROLLERS:
        t0 = time.time()
        print(f"\n=== {ds} / {c} ===", flush=True)
        r = subprocess.run([PY, "runner_2d.py", "--dataset", ds, "--controller", c],
                           cwd=HERE, capture_output=True, text=True)
        tail = "\n".join(l for l in r.stdout.splitlines()
                         if "scene" in l or "exit time" in l or "collision_ratio" in l
                         or "infeasible_ratio" in l)
        print(tail, flush=True)
        print(f"  [{ds}/{c}] rc={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
        if r.returncode != 0:
            print("  STDERR:", r.stderr[-600:], flush=True)
print("\nALL DONE", flush=True)
