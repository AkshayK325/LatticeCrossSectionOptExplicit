"""Split a generated CSV into row-preserving parts below 49,000,000 bytes."""
from pathlib import Path
import argparse

TARGET_BYTES = 49_000_000


def split_csv(source: Path, remove_original: bool = False) -> list[Path]:
    if not source.is_file():
        raise FileNotFoundError(f"CSV file not found: {source}")

    if source.stat().st_size <= TARGET_BYTES:
        return []

    for old_part in source.parent.glob(f"{source.stem}_f*.csv"):
        old_part.unlink()

    parts = []
    part_number = 0
    bytes_written = 0
    output = None

    try:
        with source.open("rb") as input_file:
            header = input_file.readline()
            if not header:
                raise ValueError(f"CSV file is empty: {source}")

            for line in input_file:
                if output is None or bytes_written >= TARGET_BYTES:
                    if output is not None:
                        output.close()

                    part_number += 1
                    destination = source.parent / f"{source.stem}_f{part_number}.csv"
                    output = destination.open("wb")
                    output.write(header)
                    bytes_written = len(header)
                    parts.append(destination)

                output.write(line)
                bytes_written += len(line)
    finally:
        if output is not None:
            output.close()

    if remove_original:
        source.unlink()

    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="CSV file to split")
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help="delete the oversized source after all parts are created",
    )
    args = parser.parse_args()

    parts = split_csv(args.csv_file, args.remove_original)
    if not parts:
        print(f"Skipped {args.csv_file}: already below 49,000,000 bytes")
        return

    print(f"Created {len(parts)} files:")
    for part in parts:
        size_mb = part.stat().st_size / (1024 * 1024)
        print(f"  {part} ({size_mb:.2f} MiB)")


if __name__ == "__main__":
    main()
