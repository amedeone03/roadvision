# Historical experiments

These scripts record the experiments that led to the current pipeline. They are not part of `python -m roadvision`, and the application does not import them.

They were archived because a later script or a module in `src/` replaced them. The Phase 5–8 reproduction scripts stay in `scripts/` so the published results can still be regenerated from their original paths.

Run a script from the repository root with the root on `PYTHONPATH`:

```bash
PYTHONPATH=. python scripts/legacy/run_video.py
```

`python scripts/legacy/<script>.py` alone does not see the `src` package. That was already true for `python scripts/<script>.py` before the move.

Most of these scripts expect a local driving video or CSV files under `runs/`. Those files are not in Git. A fresh clone can open the scripts, and it cannot rerun every experiment.
