#!/usr/bin/env python

"""Pre-generate a local Padova/PARSEC isochrone database for offline BEAST runs."""

import argparse
from pathlib import Path

from beast.physicsmodel.stars.ezpadova import parsec
from beast.physicsmodel.stars.padova_local import PadovaLocalDatabase, PadovaQuerySpec


def _parse_z_values(args):
    values = []
    if args.z is not None:
        values.extend(args.z)
    if args.z_file is not None:
        for line in Path(args.z_file).read_text(encoding="utf8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line))
    if len(values) == 0:
        raise ValueError("Provide at least one metallicity with --z or --z-file")
    return values


def generate_padova_database(
    outdir,
    z_values,
    logt_min,
    logt_max,
    dlogt,
    modeltype="parsec12s_r14",
    overwrite=False,
):
    database = PadovaLocalDatabase(outdir)
    for z in z_values:
        spec = PadovaQuerySpec.for_beast_t_isochrones(
            logt_min,
            logt_max,
            dlogt,
            z,
            modeltype=modeltype,
        )
        if database.contains(spec) and not overwrite:
            print(f"already cached z={z:g} hash={spec.key()}")
            continue

        print(f"querying z={z:g} hash={spec.key()}")
        raw_table = parsec.get_t_isochrones(
            spec.logtmin,
            spec.logtmax,
            spec.dlogt,
            spec.z,
            model=modeltype,
        )
        database.write_raw_table(
            spec,
            raw_table,
            metadata={"source": "generate_padova_database", "modeltype": modeltype},
        )
        print(f"stored z={z:g} rows={len(raw_table)} hash={spec.key()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, help="Local database directory")
    parser.add_argument("--z", nargs="*", type=float, help="Metallicity Z values")
    parser.add_argument("--z-file", help="Text file with one Z value per line")
    parser.add_argument("--logt-min", type=float, default=6.0)
    parser.add_argument("--logt-max", type=float, default=10.13)
    parser.add_argument("--dlogt", type=float, default=0.05)
    parser.add_argument("--modeltype", default="parsec12s_r14")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    generate_padova_database(
        args.outdir,
        _parse_z_values(args),
        args.logt_min,
        args.logt_max,
        args.dlogt,
        modeltype=args.modeltype,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
