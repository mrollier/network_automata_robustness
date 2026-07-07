"""Resumable HDF5 storage for rule-sweep results.

Layout per group (matches the historical results_*.h5 schema read by
manuscript/essential_metrics-figures.ipynb, plus provenance attrs):

    <group>/rules            [K, 2] int64 (beta, sigma)
    <group>/state/median     [K] float64        (likewise q1, q3)
    <group>/defect/median    [K] float64        (likewise q1, q3)
    <group>.attrs: params (json), seed, llna_version, git_commit,
                   created_at, rows_done

`rows_done` is a high-water mark: chunks are written in rule order, so an
interrupted sweep resumes at the first unwritten rule.
"""

import json
import subprocess
import time

import h5py
import numpy as np


def git_commit_or_unknown() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class SweepFile:
    """Chunk-appendable sweep results with resume support."""

    def __init__(self, path):
        self.path = str(path)

    def init_group(self, group: str, rules: np.ndarray, columns: dict[str, tuple], attrs: dict) -> int:
        """Create (or reopen) a result group.

        Parameters
        ----------
        rules : ndarray [K, 2]
        columns : dict name -> (shape_suffix tuple, dtype), e.g.
            {"state/median": ((), "f8")}
        attrs : run parameters; stored as a json attr plus scalars.

        Returns
        -------
        int
            Resume point: number of rule rows already completed.
        """
        rules = np.asarray(rules, dtype=np.int64)
        with h5py.File(self.path, "a") as f:
            if group in f:
                grp = f[group]
                if not np.array_equal(grp["rules"][...], rules):
                    raise ValueError(
                        f"Group '{group}' in {self.path} exists with a different rule table; "
                        "use a fresh output file or group."
                    )
                return int(grp.attrs["rows_done"])
            grp = f.create_group(group)
            grp.create_dataset("rules", data=rules)
            for name, (suffix, dtype) in columns.items():
                grp.create_dataset(name, shape=(len(rules), *suffix), dtype=dtype, compression="gzip")
            grp.attrs["params"] = json.dumps(attrs, default=str)
            for key, value in attrs.items():
                if isinstance(value, int | float | str | bool):
                    grp.attrs[key] = value
            grp.attrs["llna_version"] = __import__("llna").__version__
            grp.attrs["git_commit"] = git_commit_or_unknown()
            grp.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            grp.attrs["rows_done"] = 0
            return 0

    def write_rows(self, group: str, start: int, data: dict[str, np.ndarray]) -> None:
        """Write one rule chunk (rows [start, start+n)) and advance rows_done."""
        with h5py.File(self.path, "a") as f:
            grp = f[group]
            n = None
            for name, values in data.items():
                values = np.asarray(values)
                n = len(values) if n is None else n
                if len(values) != n:
                    raise ValueError("All columns in a chunk must have equal length.")
                grp[name][start : start + n] = values
            grp.attrs["rows_done"] = max(int(grp.attrs["rows_done"]), start + n)

    def read_group(self, group: str) -> tuple[dict[str, np.ndarray], dict]:
        """All datasets of a group (recursively, '/'-joined names) plus attrs."""
        out: dict[str, np.ndarray] = {}
        with h5py.File(self.path, "r") as f:
            grp = f[group]

            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    out[name] = obj[...]

            grp.visititems(collect)
            attrs = dict(grp.attrs)
        return out, attrs
