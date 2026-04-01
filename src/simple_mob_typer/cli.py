import logging
import tarfile
import tempfile
import time
from pathlib import Path

import click

from .blast import (
    BlastError,
    ensure_blast_dependencies,
    make_blast_db,
    run_blastn,
    run_tblastn,
)
from .download import DEFAULT_MARKER_DB_URL, initialize_marker_databases
from .fasta import normalize_fasta_and_collect_metadata
from .typing import (
    DatabasePaths,
    Thresholds,
    classify_records_query,
    create_biomarker_report_query,
    format_classification_output_query,
    prepare_report_frames,
    prepare_typing_frames,
    read_biomarker_frames,
)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "show_default": True,
}
LOGGER = logging.getLogger("simple_mob_typer")
LOGGER_SOURCE_LABELS = {
    "simple_mob_typer": "run",
    "simple_mob_typer.blast": "blast",
}


class SimpleMobTyperLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        level = f"{record.levelname:<5}"
        source = f"{LOGGER_SOURCE_LABELS.get(record.name, record.name):<5}"
        message = record.getMessage()
        formatted = f"{timestamp} | {level} | {source} | {message}"
        if record.exc_info:
            formatted = f"{formatted}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            formatted = f"{formatted}\n{self.formatStack(record.stack_info)}"
        return formatted


def configure_logging(debug: bool) -> None:
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(SimpleMobTyperLogFormatter())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)


def resolve_databases(
    *,
    database_dir: Path | None,
    replicon_db: Path | None,
    relaxase_db: Path | None,
    mpf_db: Path | None,
    orit_db: Path | None,
) -> DatabasePaths:
    resolved_database_dir = database_dir.resolve() if database_dir else None

    def resolve_path(explicit_path: Path | None, filename: str) -> Path:
        if explicit_path is not None:
            return explicit_path.resolve()
        if resolved_database_dir is not None:
            return (resolved_database_dir / filename).resolve()
        raise click.ClickException(
            f"Missing database path for {filename}. Set --database-dir or the explicit --*-db path."
        )

    paths = DatabasePaths(
        replicon=resolve_path(replicon_db, "rep.dna.fas"),
        relaxase=resolve_path(relaxase_db, "mob.proteins.faa"),
        mpf=resolve_path(mpf_db, "mpf.proteins.faa"),
        orit=resolve_path(orit_db, "orit.fas"),
    )
    for path in (paths.replicon, paths.relaxase, paths.mpf, paths.orit):
        if not path.is_file():
            raise click.ClickException(f"Required database file is missing: {path}")
    return paths


def build_thresholds(
    *,
    min_rep_ident: float,
    min_rep_cov: float,
    min_rep_evalue: float,
    min_mob_ident: float,
    min_mob_cov: float,
    min_mob_evalue: float,
) -> Thresholds:
    return Thresholds(
        min_rep_ident=min_rep_ident,
        min_rep_cov=min_rep_cov,
        min_rep_evalue=min_rep_evalue,
        min_mob_ident=min_mob_ident,
        min_mob_cov=min_mob_cov,
        min_mob_evalue=min_mob_evalue,
    )


def run_typing(
    *,
    infile: Path,
    out_file: Path,
    database_dir: Path | None,
    replicon_db: Path | None,
    relaxase_db: Path | None,
    mpf_db: Path | None,
    orit_db: Path | None,
    threads: int,
    biomarker_report_file: Path | None,
    min_rep_ident: float,
    min_rep_cov: float,
    min_rep_evalue: float,
    min_mob_ident: float,
    min_mob_cov: float,
    min_mob_evalue: float,
    debug: bool,
) -> None:
    run_started_at = time.perf_counter()
    configure_logging(debug)

    input_path = infile.resolve()
    if not input_path.is_file():
        raise click.ClickException(f"Input FASTA does not exist: {input_path}")

    LOGGER.info("Starting plasmid typing for %s", input_path)
    thresholds = build_thresholds(
        min_rep_ident=min_rep_ident,
        min_rep_cov=min_rep_cov,
        min_rep_evalue=min_rep_evalue,
        min_mob_ident=min_mob_ident,
        min_mob_cov=min_mob_cov,
        min_mob_evalue=min_mob_evalue,
    )
    databases = resolve_databases(
        database_dir=database_dir,
        replicon_db=replicon_db,
        relaxase_db=relaxase_db,
        mpf_db=mpf_db,
        orit_db=orit_db,
    )
    LOGGER.debug("Resolved database paths: %s", databases)
    ensure_blast_dependencies()
    output_path = out_file.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Results output path: %s", output_path)

    with tempfile.TemporaryDirectory(prefix="simple_mob_typer_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        LOGGER.debug("Temporary workspace: %s", temp_dir_path)
        normalized_fasta = temp_dir_path / "normalized.fasta"
        LOGGER.info("Parsing FASTA input")
        try:
            records = normalize_fasta_and_collect_metadata(input_path, normalized_fasta)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if not records:
            raise click.ClickException(f"No FASTA records found in {input_path}")
        LOGGER.info("Parsed %d FASTA records", len(records))
        make_blast_db(normalized_fasta, "nucl", label="normalized input FASTA")

        replicon_output = temp_dir_path / "replicon.tsv"
        relaxase_output = temp_dir_path / "relaxase.tsv"
        mpf_output = temp_dir_path / "mpf.tsv"
        orit_output = temp_dir_path / "orit.tsv"

        try:
            LOGGER.info("Running biomarker searches")
            run_blastn(
                databases.replicon,
                normalized_fasta,
                replicon_output,
                min_ident=thresholds.min_rep_ident,
                evalue=thresholds.min_rep_evalue,
                threads=threads,
                label="replicon markers",
            )
            run_tblastn(
                databases.relaxase,
                normalized_fasta,
                relaxase_output,
                evalue=thresholds.min_mob_evalue,
                threads=threads,
                label="relaxase markers",
            )
            run_tblastn(
                databases.mpf,
                normalized_fasta,
                mpf_output,
                evalue=thresholds.min_mob_evalue,
                threads=threads,
                label="MPF markers",
            )
            run_blastn(
                databases.orit,
                normalized_fasta,
                orit_output,
                min_ident=thresholds.min_rep_ident,
                evalue=thresholds.min_rep_evalue,
                threads=threads,
                label="oriT markers",
            )
        except BlastError as exc:
            raise click.ClickException(str(exc)) from exc

        LOGGER.info("Preparing biomarker hits")
        raw_frames = read_biomarker_frames(
            replicon_output=replicon_output,
            relaxase_output=relaxase_output,
            mpf_output=mpf_output,
            orit_output=orit_output,
        )
        prepared_frames = prepare_typing_frames(raw_frames, thresholds)
        LOGGER.info(
            "Prepared hits: replicon=%d relaxase=%d mpf=%d oriT=%d",
            prepared_frames.replicon.height,
            prepared_frames.relaxase.height,
            prepared_frames.mate_pair_formation.height,
            prepared_frames.orit.height,
        )

        LOGGER.info("Classifying plasmid records")
        classification_query = classify_records_query(
            records,
            prepared_frames=prepared_frames,
        )
        LOGGER.info("Writing classification results")
        format_classification_output_query(classification_query).sink_csv(
            output_path, separator="\t"
        )

        if biomarker_report_file is not None:
            biomarker_output = biomarker_report_file.resolve()
            biomarker_output.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Writing biomarker report to %s", biomarker_output)
            prepared_report_frames = prepare_report_frames(raw_frames, thresholds)
            create_biomarker_report_query(
                prepared_report_frames=prepared_report_frames,
            ).sink_csv(biomarker_output, separator="\t")

        LOGGER.info(
            "Wrote %d plasmid classifications in %.1fs",
            len(records),
            time.perf_counter() - run_started_at,
        )


@click.group(context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """Simple MOB-typer-compatible plasmid classifier."""


@cli.command("run", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--infile",
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Input FASTA containing one or more plasmids",
)
@click.option(
    "--out-file",
    "out_file",
    required=True,
    type=click.Path(
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Output TSV path",
)
@click.option(
    "--database-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory containing rep.dna.fas, mob.proteins.faa, mpf.proteins.faa, and orit.fas",
)
@click.option(
    "--replicon-db",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Explicit path to rep.dna.fas",
)
@click.option(
    "--relaxase-db",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Explicit path to mob.proteins.faa",
)
@click.option(
    "--mpf-db",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Explicit path to mpf.proteins.faa",
)
@click.option(
    "--orit-db",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Explicit path to orit.fas",
)
@click.option(
    "--threads",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of BLAST threads",
)
@click.option(
    "--biomarker-report-file",
    type=click.Path(
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Optional output path for retained biomarker hits",
)
@click.option(
    "--min-rep-ident",
    type=click.FloatRange(min=0.0, max=100.0),
    default=80.0,
    show_default=True,
    help="Minimum replicon percent identity",
)
@click.option(
    "--min-rep-cov",
    type=click.FloatRange(min=0.0, max=100.0),
    default=80.0,
    show_default=True,
    help="Minimum replicon query coverage percentage",
)
@click.option(
    "--min-rep-evalue",
    type=click.FloatRange(min=0.0),
    default=1e-5,
    show_default=True,
    help="Maximum replicon E-value",
)
@click.option(
    "--min-mob-ident",
    type=click.FloatRange(min=0.0, max=100.0),
    default=80.0,
    show_default=True,
    help="Minimum relaxase and MPF percent identity",
)
@click.option(
    "--min-mob-cov",
    type=click.FloatRange(min=0.0, max=100.0),
    default=80.0,
    show_default=True,
    help="Minimum relaxase and MPF query coverage percentage",
)
@click.option(
    "--min-mob-evalue",
    type=click.FloatRange(min=0.0),
    default=1e-5,
    show_default=True,
    help="Maximum relaxase and MPF E-value",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
def run_command(
    infile: Path,
    out_file: Path,
    database_dir: Path | None,
    replicon_db: Path | None,
    relaxase_db: Path | None,
    mpf_db: Path | None,
    orit_db: Path | None,
    threads: int,
    biomarker_report_file: Path | None,
    min_rep_ident: float,
    min_rep_cov: float,
    min_rep_evalue: float,
    min_mob_ident: float,
    min_mob_cov: float,
    min_mob_evalue: float,
    debug: bool,
) -> None:
    """Run plasmid typing on an input FASTA."""
    run_typing(
        infile=infile,
        out_file=out_file,
        database_dir=database_dir,
        replicon_db=replicon_db,
        relaxase_db=relaxase_db,
        mpf_db=mpf_db,
        orit_db=orit_db,
        threads=threads,
        biomarker_report_file=biomarker_report_file,
        min_rep_ident=min_rep_ident,
        min_rep_cov=min_rep_cov,
        min_rep_evalue=min_rep_evalue,
        min_mob_ident=min_mob_ident,
        min_mob_cov=min_mob_cov,
        min_mob_evalue=min_mob_evalue,
        debug=debug,
    )


@cli.command("download", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--database-dir",
    required=True,
    type=click.Path(
        file_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Directory to populate with rep.dna.fas, mob.proteins.faa, mpf.proteins.faa, and orit.fas",
)
@click.option(
    "--url",
    default=DEFAULT_MARKER_DB_URL,
    show_default=True,
    help="Archive URL containing the official MOB-suite database bundle",
)
@click.option(
    "--archive-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Use an existing local .tar.gz archive instead of downloading from --url",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing marker database files in --database-dir",
)
@click.option(
    "--keep-archive",
    is_flag=True,
    help="Retain the downloaded archive next to the extracted files",
)
def download_command(
    database_dir: Path,
    url: str,
    archive_path: Path | None,
    force: bool,
    keep_archive: bool,
) -> None:
    """Download the marker FASTA databases used by the typer."""
    resolved_archive_path = archive_path.resolve() if archive_path else None
    try:
        extracted = initialize_marker_databases(
            database_dir,
            url=url,
            archive_path=resolved_archive_path,
            force=force,
            keep_archive=keep_archive,
        )
    except (OSError, tarfile.TarError) as exc:
        raise click.ClickException(str(exc)) from exc

    for path in extracted:
        click.echo(str(path))


def main() -> None:
    cli()
