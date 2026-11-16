
import argparse
import csv
import hashlib
import io
import posixpath
from datetime import datetime
from decimal import Decimal, InvalidOperation
from collections import Counter
from pathlib import Path


LABEL_MAP = {
    "cargo": "Cargo", "tanker": "Tanker", "passengership": "Passenger",
    "tug": "Tug/Work", "dredger": "Tug/Work", "rescue": "Tug/Work",
    "pilotvessel": "Tug/Work", "fishing": "Fishing",
    "pleasurecraft": "Small craft", "sailing": "Small craft",
    "background": "Background", "other": "Other",
}
EXPECTED = {"Cargo": 97, "Tanker": 5, "Passenger": 71, "Tug/Work": 202,
            "Fishing": 34, "Small craft": 38, "Background": 38, "Other": 40}
SPLITS = ("train", "val", "test")
TARGET = (367, 79, 79)
QUOTAS = {
    "Cargo": (68, 14, 15), "Tanker": (3, 1, 1),
    "Passenger": (50, 11, 10), "Tug/Work": (142, 30, 30),
    "Fishing": (24, 5, 5), "Small craft": (26, 6, 6),
    "Background": (26, 6, 6), "Other": (28, 6, 6),
}
COLUMNS = ["recording_id", "path", "dataset", "unified_label", "split", "duration_sec"]
SOURCE_COLUMNS = ["path", "label", "duration_sec", "utc_begin", "utc_end"]
DATASET = "VTUAD"



def require(condition, message):
    if not condition:
        raise ValueError(message)


def recording_id(path):
    require(bool(path.strip()), "Empty WAV path")
    normalized = posixpath.normpath(path.strip().replace("\\", "/"))
    require(normalized.lower().endswith(".wav"), f"Not a WAV path: {path!r}")
    return normalized[:-4]


def read_metadata(source):
    raw = source.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    require(len(fields) == len(set(fields)), "Duplicate CSV column names")
    require(set(SOURCE_COLUMNS) <= set(fields),
            f"Missing required columns: {sorted(set(SOURCE_COLUMNS) - set(fields))}")
    rows = list(reader)
    seen = set()
    for number, row in enumerate(rows, 2):
        require(None not in row and all(v is not None for v in row.values()),
                f"Malformed CSV row {number}")
        require(row["label"] in LABEL_MAP,
                f"Unknown label at row {number}: {row['label']!r}")
        rid = recording_id(row["path"])
        # Conservative case-insensitive comparison also catches Windows aliases.
        key = rid.casefold()
        require(key not in seen, f"Duplicate WAV at row {number}: {rid}")
        seen.add(key)
        for column, expected in (("recording_id", rid),
                                 ("unified_label", LABEL_MAP[row["label"]])):
            require(not row.get(column) or row[column] == expected,
                    f"Conflicting {column} at row {number}")
    counts = Counter(LABEL_MAP[row["label"]] for row in rows)
    require(counts == Counter(EXPECTED),
            f"Expected 525 VTUAD recordings with counts {EXPECTED}; got {dict(counts)}")
    return raw, fields, rows


def correct_durations(rows, enabled=True):
    """Return overrides for the observed +300 s anomaly, without editing metadata.

    A timestamp-derived value is provisional until checked against WAV headers.
    Small (1-2 s) metadata/timestamp differences are intentionally untouched.
    """
    fixes = {}
    if not enabled:
        return fixes
    for number, row in enumerate(rows, 2):
        try:
            declared = Decimal(row["duration_sec"])
            begin = datetime.fromisoformat(row["utc_begin"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(row["utc_end"].replace("Z", "+00:00"))
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"Invalid duration or UTC timestamp at row {number}") from error
        require(declared.is_finite() and declared > 0,
                f"Invalid duration at row {number}: {row['duration_sec']!r}")
        require(begin.utcoffset() is not None and end.utcoffset() is not None,
                f"Timezone missing at row {number}")
        elapsed = Decimal(str((end - begin).total_seconds()))
        require(elapsed > 0, f"Non-positive UTC interval at row {number}")
        if declared - elapsed == Decimal("300"):
            # Keep the original CSV untouched; catalog only gets corrected seconds.
            fixes[recording_id(row["path"])] = str(elapsed)
    return fixes


def allocate_counts():
    """Validate both row and column sums before using the fixed quotas."""
    require(set(QUOTAS) == set(EXPECTED), "Quota labels differ")
    require(all(len(q) == 3 and all(type(n) is int and n > 0 for n in q)
                and sum(q) == EXPECTED[c] for c, q in QUOTAS.items()),
            "Invalid class quotas")
    require(tuple(sum(q[j] for q in QUOTAS.values()) for j in range(3)) == TARGET,
            "Global quota error")
    require(QUOTAS["Tanker"] == (3, 1, 1), "Tanker quota error")
    return QUOTAS


def make_catalog(rows, seed, duration_fixes=None):
    duration_fixes = duration_fixes or {}
    quotas = allocate_counts()
    assignment = {}
    for label in EXPECTED:
        ids = sorted(recording_id(r["path"]) for r in rows
                     if LABEL_MAP[r["label"]] == label)
        ids.sort(key=lambda rid: (hashlib.sha256(
            f"{seed}:{rid}".encode("utf-8")).digest(), rid))
        offset = 0
        for split, count in zip(SPLITS, quotas[label]):
            for rid in ids[offset:offset + count]:
                assignment[rid] = split
            offset += count
    return [dict(path=row["path"],
                 duration_sec=duration_fixes.get(recording_id(row["path"]),
                                                 row["duration_sec"]),
                 dataset=DATASET, recording_id=recording_id(row["path"]),
                 unified_label=LABEL_MAP[row["label"]],
                 split=assignment[recording_id(row["path"])]) for row in rows]


def validate(original, catalog, duration_fixes=None):
    duration_fixes = duration_fixes or {}
    require(len(catalog) == len(original) == 525, "Missing or extra rows")
    ids = [r["recording_id"].casefold() for r in catalog]
    require(len(set(ids)) == len(ids), "Duplicate recording IDs")
    require(all(set(r) == set(COLUMNS) for r in catalog), "Unexpected output columns")
    wav_sets = [{recording_id(r["path"]).casefold() for r in catalog
                 if r["split"] == s} for s in SPLITS]
    require(all(not wav_sets[i] & wav_sets[j]
                for i in range(3) for j in range(i + 1, 3)), "WAV leakage")
    counts = Counter()
    for old, new in zip(original, catalog):
        require(new["recording_id"] == recording_id(old["path"]), "Recording ID error")
        require(new["unified_label"] == LABEL_MAP[old["label"]], "Label mapping error")
        require(new["dataset"] == DATASET, "Dataset name error")
        require(new["split"] in SPLITS, "Missing or invalid split")
        require(new["path"] == old["path"], "Original path changed")
        require(new["duration_sec"] == duration_fixes.get(
            recording_id(old["path"]), old["duration_sec"]),
            "Duration correction mismatch")
        counts[new["unified_label"], new["split"]] += 1
    for label, quotas in allocate_counts().items():
        require(tuple(counts[label, s] for s in SPLITS) == quotas, f"Quota error: {label}")
    require(tuple(sum(counts[c, s] for c in EXPECTED) for s in SPLITS) == TARGET,
            "Incorrect train/val/test totals")


def validate_wavs(catalog, wav_root):
    """Check real files, resolved aliases, hardlinks, and WAV coverage."""
    root = wav_root.resolve(strict=True)
    seen = set()
    listed = set()
    for row in catalog:
        path = (root / row["path"].replace("\\", "/")).resolve(strict=True)
        require(path.is_relative_to(root), f"WAV outside dataset root: {path}")
        require(path.is_file(), f"Not a file: {path}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        require(identity not in seen, f"Duplicate physical WAV: {path}")
        seen.add(identity)
        listed.add(path)
    actual = {p.resolve() for p in root.rglob("*")
              if p.is_file() and p.suffix.lower() == ".wav"}
    require(listed == actual,
            f"WAV coverage mismatch: unlisted={len(actual-listed)}, missing={len(listed-actual)}")
    print(f"PASS: {len(listed)} real WAVs; full coverage; no physical-file aliases")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata.csv")
    parser.add_argument("--output", type=Path, default=root / "data/catalog.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-duration-fix", action="store_true",
                        help="Keep source duration_sec values unchanged")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly allow replacing an existing output catalog")
    parser.add_argument("--wav-root", type=Path,
                        help="Dataset root containing vessel/ and background/; verifies actual WAVs")
    args = parser.parse_args()
    require(args.metadata.resolve() != args.output.resolve(), "Cannot overwrite metadata")
    if args.output.exists():
        require(not args.output.samefile(args.metadata), "Output aliases metadata")
    if args.output.exists():
        require(args.overwrite,
                f"Output already exists: {args.output} (use --overwrite to replace)")
    raw, fields, rows = read_metadata(args.metadata)
    duration_fixes = correct_durations(rows, enabled=not args.no_duration_fix)
    catalogs = [make_catalog(rows, args.seed, duration_fixes) for _ in range(3)]
    for catalog in catalogs:
        validate(rows, catalog, duration_fixes)
    require(catalogs[0] == catalogs[1] == catalogs[2], "Three-run reproducibility failed")
    require(args.metadata.read_bytes() == raw, "Metadata changed during processing")
    if args.wav_root is not None:
        validate_wavs(catalogs[0], args.wav_root)
    columns = COLUMNS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(catalogs[0])
    with args.output.open(encoding="utf-8", newline="") as stream:
        validate(rows, list(csv.DictReader(stream)), duration_fixes)
    require(args.metadata.read_bytes() == raw, "Source metadata was modified")
    print(f"Duration corrected using UTC (+300 s anomaly): {len(duration_fixes)} recordings")
    if duration_fixes:
        print("NOTICE: UTC-based durations are provisional until checked against WAV lengths")
    print("Class           train  val test total")
    for label, quotas in allocate_counts().items():
        print(f"{label:15} {quotas[0]:5} {quotas[1]:4} {quotas[2]:4} {sum(quotas):5}")
    print("TOTAL             367   79   79   525")
    print(f"PASS: three identical runs; mapping, uniqueness, coverage, source unchanged")
    print(f"Source SHA256: {hashlib.sha256(raw).hexdigest()}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, UnicodeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
