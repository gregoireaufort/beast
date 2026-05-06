"""Local Padova/PARSEC isochrone database helpers."""

import gzip
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import astropy.table

from beast.physicsmodel.stars.ezpadova import parsec


class MissingIsochroneError(FileNotFoundError):
    """Raised when an offline Padova database does not contain a query."""


def _canonical_value(value):
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_value(value[key]) for key in sorted(value)}
    return value


@dataclass(frozen=True)
class PadovaQuerySpec:
    """Canonical description of one Padova/CMD query used by BEAST."""

    modeltype: str
    logtmin: float
    logtmax: float
    dlogt: float
    z: float
    filterPMS: bool = False
    filterBad: bool = False
    query_args: dict = field(default_factory=dict)

    @classmethod
    def for_beast_t_isochrones(
        cls,
        logtmin,
        logtmax,
        dlogt,
        z,
        modeltype="parsec12s_r14",
        filterPMS=False,
        filterBad=False,
        query_args=None,
    ):
        if query_args is None:
            query_args = parsec.get_ezpadova_args(model=modeltype)
        return cls(
            modeltype=modeltype,
            logtmin=float(max(6.0, logtmin)),
            logtmax=float(min(10.13, logtmax)),
            dlogt=float(dlogt),
            z=float(z),
            filterPMS=bool(filterPMS),
            filterBad=bool(filterBad),
            query_args=dict(query_args),
        )

    def as_dict(self):
        return {
            "modeltype": self.modeltype,
            "logtmin": self.logtmin,
            "logtmax": self.logtmax,
            "dlogt": self.dlogt,
            "z": self.z,
            "filterPMS": self.filterPMS,
            "filterBad": self.filterBad,
            "query_args": self.query_args,
        }

    def canonical_json(self):
        return json.dumps(
            _canonical_value(self.as_dict()),
            sort_keys=True,
            separators=(",", ":"),
        )

    def key(self):
        return hashlib.sha256(self.canonical_json().encode("utf8")).hexdigest()


class PadovaLocalDatabase:
    """File-backed database for raw Padova/CMD query results."""

    index_filename = "index.json"

    def __init__(self, root):
        self.root = Path(root)
        self.index_path = self.root / self.index_filename

    def _load_index(self):
        if not self.index_path.exists():
            return {"version": 1, "entries": {}}
        with self.index_path.open("r", encoding="utf8") as fp:
            return json.load(fp)

    def _write_index(self, index):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf8") as fp:
            json.dump(index, fp, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp_path, self.index_path)

    def paths_for(self, spec):
        key = spec.key()
        prefix = key[:2]
        base = self.root / "queries" / prefix / key
        return {
            "base": base,
            "table": base / "raw.ecsv",
            "raw": base / "raw.dat.gz",
            "metadata": base / "metadata.json",
        }

    def contains(self, spec):
        paths = self.paths_for(spec)
        return paths["table"].exists()

    def write_raw_table(self, spec, table, raw_bytes=None, metadata=None):
        paths = self.paths_for(spec)
        paths["base"].mkdir(parents=True, exist_ok=True)

        astropy.table.Table(table).write(
            paths["table"], format="ascii.ecsv", overwrite=True
        )

        raw_sha256 = None
        if raw_bytes is not None:
            raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            with gzip.open(paths["raw"], "wb") as fp:
                fp.write(raw_bytes)

        entry_metadata = {
            "query_hash": spec.key(),
            "canonical_query": json.loads(spec.canonical_json()),
            "raw_sha256": raw_sha256,
            "table_path": str(paths["table"].relative_to(self.root)),
            "raw_path": str(paths["raw"].relative_to(self.root))
            if raw_bytes is not None
            else None,
            "n_rows": len(table),
            "columns": list(table.colnames),
        }
        if metadata is not None:
            entry_metadata["metadata"] = metadata

        with paths["metadata"].open("w", encoding="utf8") as fp:
            json.dump(entry_metadata, fp, indent=2, sort_keys=True)
            fp.write("\n")

        index = self._load_index()
        index["entries"][spec.key()] = {
            "canonical_query": entry_metadata["canonical_query"],
            "metadata_path": str(paths["metadata"].relative_to(self.root)),
            "table_path": entry_metadata["table_path"],
            "n_rows": len(table),
        }
        self._write_index(index)

    def read_raw_table(self, spec):
        paths = self.paths_for(spec)
        if not paths["table"].exists():
            index = self._load_index()
            available = len(index.get("entries", {}))
            raise MissingIsochroneError(
                "Padova local database miss for query "
                f"{spec.canonical_json()} (hash={spec.key()}). "
                f"Database has {available} entries at {self.root}."
            )
        return astropy.table.Table.read(paths["table"], format="ascii.ecsv")
