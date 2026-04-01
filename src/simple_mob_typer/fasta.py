import sys
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path

if sys.version_info >= (3, 14):
    from compression import bz2, gzip, lzma, zstd
else:
    import bz2
    import gzip
    import lzma

    zstd = None


@dataclass(frozen=True, slots=True)
class RecordMetadata:
    accession: str
    size: int
    gc: float
    sequence_md5: str


def _iter_fasta_entries(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("rb") as handle:
        signature = handle.peek(8)[:8]

    if tuple(signature[:2]) == (0x1F, 0x8B):
        stream = gzip.open(path, "rt", encoding="utf-8")
    elif tuple(signature[:3]) == (0x42, 0x5A, 0x68):
        stream = bz2.open(path, "rt", encoding="utf-8")
    elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
        stream = lzma.open(path, "rt", encoding="utf-8")
    elif tuple(signature[:4]) == (0x28, 0xB5, 0x2F, 0xFD):
        if zstd is None:
            raise ValueError("Zstandard-compressed FASTA requires Python 3.14+")
        stream = zstd.open(path, "rt", encoding="utf-8")
    else:
        stream = path.open("r", encoding="utf-8")

    with stream as handle:
        accession = None
        sequence_lines: list[str] = []

        for raw_line in handle:
            line = raw_line.removesuffix("\n")
            if line.startswith(">"):
                if accession is not None:
                    yield accession, "".join(sequence_lines).upper()
                accession = line[1:].split(None, 1)[0].strip()
                sequence_lines = []
                continue
            if accession is None:
                continue
            sequence_line = line.replace(" ", "")
            if sequence_line:
                sequence_lines.append(sequence_line)

        if accession is not None:
            yield accession, "".join(sequence_lines).upper()


def normalize_fasta_and_collect_metadata(
    input_path: Path,
    output_path: Path,
) -> list[RecordMetadata]:
    records: list[RecordMetadata] = []
    seen_accessions: set[str] = set()
    duplicate_accessions: set[str] = set()

    with output_path.open("w", encoding="utf-8") as output_handle:
        for accession, sequence in _iter_fasta_entries(input_path):
            if accession in seen_accessions:
                duplicate_accessions.add(accession)
            seen_accessions.add(accession)
            output_handle.write(f">{accession}\n{sequence}\n")

            size = len(sequence)
            gc = 0.0
            if sequence:
                gc = (sequence.count("G") + sequence.count("C")) / size

            records.append(
                RecordMetadata(
                    accession=accession,
                    size=size,
                    gc=gc,
                    sequence_md5=md5(sequence.encode("ascii")).hexdigest(),
                )
            )

    if duplicate_accessions:
        duplicates = ", ".join(sorted(duplicate_accessions))
        raise ValueError(f"Duplicate FASTA accessions are not allowed: {duplicates}")

    return records
