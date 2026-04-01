import logging
import shutil
import subprocess
import time
from pathlib import Path

import polars as pl


BLAST_COLUMNS = [
    "qseqid",
    "sseqid",
    "qlen",
    "slen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "length",
    "pident",
    "qcovhsp",
    "qcovs",
    "evalue",
    "bitscore",
]


BLAST_SCHEMA = {
    "qseqid": pl.Utf8,
    "sseqid": pl.Utf8,
    "qlen": pl.Int64,
    "slen": pl.Int64,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "sstart": pl.Int64,
    "send": pl.Int64,
    "length": pl.Int64,
    "pident": pl.Float64,
    "qcovhsp": pl.Float64,
    "qcovs": pl.Float64,
    "evalue": pl.Float64,
    "bitscore": pl.Float64,
}


LOGGER = logging.getLogger("simple_mob_typer.blast")


class BlastError(RuntimeError):
    """Raised when a BLAST command fails."""


def ensure_blast_dependencies() -> None:
    LOGGER.info("Checking BLAST dependencies")
    for binary in ("blastn", "tblastn", "makeblastdb"):
        if shutil.which(binary) is None:
            raise BlastError(f"Required BLAST+ binary is unavailable: {binary}")
    LOGGER.info("BLAST dependencies are available")


def _run_blast_command(
    command: list[str], *, description: str, error_message: str
) -> None:
    LOGGER.info("Running %s", description)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Command: %s", " ".join(command))
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started_at
    if completed.returncode != 0:
        LOGGER.error("Failed %s after %.1fs", description, elapsed)
        raise BlastError(completed.stderr.strip() or error_message)
    LOGGER.info("Finished %s in %.1fs", description, elapsed)


def make_blast_db(fasta_path: Path, dbtype: str, *, label: str | None = None) -> None:
    description = f"makeblastdb for {label}" if label else f"makeblastdb for {fasta_path}"
    _run_blast_command(
        ["makeblastdb", "-in", str(fasta_path), "-dbtype", dbtype],
        description=description,
        error_message="makeblastdb failed",
    )


def run_blastn(
    query_path: Path,
    db_path: Path,
    output_path: Path,
    *,
    min_ident: float,
    evalue: float,
    threads: int,
    label: str | None = None,
) -> None:
    description = f"blastn for {label}" if label else f"blastn for {query_path.name}"
    _run_blast_command(
        [
            "blastn",
            "-task",
            "megablast",
            "-query",
            str(query_path),
            "-db",
            str(db_path),
            "-num_threads",
            str(threads),
            "-evalue",
            str(evalue),
            "-dust",
            "yes",
            "-perc_identity",
            str(min_ident),
            "-max_target_seqs",
            "100000000",
            "-out",
            str(output_path),
            "-outfmt",
            "6 " + " ".join(BLAST_COLUMNS),
        ],
        description=description,
        error_message="blastn failed",
    )


def run_tblastn(
    query_path: Path,
    db_path: Path,
    output_path: Path,
    *,
    evalue: float,
    threads: int,
    label: str | None = None,
) -> None:
    description = f"tblastn for {label}" if label else f"tblastn for {query_path.name}"
    _run_blast_command(
        [
            "tblastn",
            "-query",
            str(query_path),
            "-num_threads",
            str(threads),
            "-db",
            str(db_path),
            "-evalue",
            str(evalue),
            "-out",
            str(output_path),
            "-max_target_seqs",
            "100000000",
            "-outfmt",
            "6 " + " ".join(BLAST_COLUMNS),
        ],
        description=description,
        error_message="tblastn failed",
    )


def scan_blast_table(path: Path) -> pl.LazyFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(schema=BLAST_SCHEMA).lazy()

    return pl.scan_csv(
        path,
        separator="\t",
        has_header=False,
        schema=BLAST_SCHEMA,
    )
