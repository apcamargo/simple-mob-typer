from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl

from .blast import BLAST_COLUMNS, BLAST_SCHEMA, scan_blast_table
from .fasta import RecordMetadata

ROW_INDEX_COLUMN = "row_idx"
HIT_ORDER_COLUMN = "hit_order"
EDGE_SIDE_COLUMN = "edge_side"
EDGE_DROP_COLUMN = "drop_edge_side"
RECORD_ORDER_COLUMN = "record_order"

BIOMARKER_SPECS = (
    ("replicon", "replicon"),
    ("relaxase", "relaxase"),
    ("mate-pair-formation", "mate_pair_formation"),
    ("oriT", "orit"),
)

CLASSIFICATION_SCHEMA = {
    "sample_id": pl.Utf8,
    "num_contigs": pl.Int64,
    "size": pl.Int64,
    "gc": pl.Float64,
    "md5": pl.Utf8,
    "rep_types": pl.List(pl.Utf8),
    "rep_accessions": pl.List(pl.Utf8),
    "relaxase_types": pl.List(pl.Utf8),
    "relaxase_accessions": pl.List(pl.Utf8),
    "mpf_type": pl.Utf8,
    "mpf_accessions": pl.List(pl.Utf8),
    "orit_types": pl.List(pl.Utf8),
    "orit_accessions": pl.List(pl.Utf8),
    "predicted_mobility": pl.Utf8,
}


@dataclass(frozen=True, slots=True)
class Thresholds:
    min_rep_ident: float = 80.0
    min_rep_cov: float = 80.0
    min_rep_evalue: float = 1e-5
    min_rep_length: int = 80
    min_rep_hsp_cov: float = 30.0
    min_mob_ident: float = 80.0
    min_mob_cov: float = 80.0
    min_mob_evalue: float = 1e-5
    min_mob_hsp_cov: float = 25.0
    min_orit_length: int = 80
    min_orit_hsp_cov: float = 1.0
    max_blast_query_length: int = 400000
    overlap_tolerance: int = 5


@dataclass(frozen=True, slots=True)
class DatabasePaths:
    replicon: Path
    relaxase: Path
    mpf: Path
    orit: Path


@dataclass(frozen=True, slots=True)
class _FrameThresholds:
    min_length: int
    min_cov: float
    min_hsp_cov: float
    min_ident: float
    evalue: float
    max_query_length: int | None = None


@dataclass(frozen=True, slots=True)
class RawBiomarkerFrames:
    replicon: pl.LazyFrame
    relaxase: pl.LazyFrame
    mate_pair_formation: pl.LazyFrame
    orit: pl.LazyFrame


@dataclass(frozen=True, slots=True)
class PreparedBiomarkerFrames:
    replicon: pl.DataFrame
    relaxase: pl.DataFrame
    mate_pair_formation: pl.DataFrame
    orit: pl.DataFrame


def _empty_biomarker_report_query() -> pl.LazyFrame:
    return (
        pl.DataFrame(schema={**BLAST_SCHEMA, "biomarker": pl.Utf8})
        .select([*BLAST_COLUMNS, "biomarker"])
        .lazy()
    )


def _normalize_coordinates(
    frame: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    return frame.with_columns(
        [
            pl.when(pl.col("qstart") <= pl.col("qend"))
            .then(pl.col("qstart"))
            .otherwise(pl.col("qend"))
            .alias("qstart"),
            pl.when(pl.col("qstart") <= pl.col("qend"))
            .then(pl.col("qend"))
            .otherwise(pl.col("qstart"))
            .alias("qend"),
            pl.when(pl.col("sstart") <= pl.col("send"))
            .then(pl.col("sstart"))
            .otherwise(pl.col("send"))
            .alias("sstart"),
            pl.when(pl.col("sstart") <= pl.col("send"))
            .then(pl.col("send"))
            .otherwise(pl.col("sstart"))
            .alias("send"),
        ]
    )


def _apply_thresholds(
    frame: pl.DataFrame | pl.LazyFrame,
    thresholds: _FrameThresholds,
) -> pl.DataFrame | pl.LazyFrame:
    filtered = (
        _normalize_coordinates(frame)
        .filter(pl.col("length") >= thresholds.min_length)
        .filter(pl.col("qcovs") >= thresholds.min_cov)
        .filter(pl.col("qcovhsp") >= thresholds.min_hsp_cov)
        .filter(pl.col("pident") >= thresholds.min_ident)
        .filter(pl.col("evalue") <= thresholds.evalue)
    )
    if thresholds.max_query_length is not None:
        filtered = filtered.filter(pl.col("qlen") <= thresholds.max_query_length)
    return filtered


def _remove_split_edge_hits(frame: pl.LazyFrame) -> pl.LazyFrame:
    annotated = frame.with_columns(
        pl.when(pl.col("sstart") == 1)
        .then(pl.lit("start"))
        .when(pl.col("send") == pl.col("slen"))
        .then(pl.lit("end"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias(EDGE_SIDE_COLUMN)
    )

    edge_decisions = (
        annotated.filter(pl.col(EDGE_SIDE_COLUMN).is_not_null())
        .group_by(["qseqid", "sseqid"])
        .agg(
            pl.when(pl.col(EDGE_SIDE_COLUMN) == "start")
            .then(pl.col("bitscore"))
            .max()
            .alias("__start_max_bitscore"),
            pl.when(pl.col(EDGE_SIDE_COLUMN) == "end")
            .then(pl.col("bitscore"))
            .max()
            .alias("__end_max_bitscore"),
        )
        .with_columns(
            pl.when(
                pl.col("__start_max_bitscore").is_not_null()
                & pl.col("__end_max_bitscore").is_not_null()
            )
            .then(
                pl.when(
                    pl.col("__start_max_bitscore") >= pl.col("__end_max_bitscore")
                )
                .then(pl.lit("start"))
                .otherwise(pl.lit("end"))
            )
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(EDGE_DROP_COLUMN)
        )
        .select(["qseqid", "sseqid", EDGE_DROP_COLUMN])
    )

    drop_mask = (
        pl.col(EDGE_SIDE_COLUMN).is_not_null()
        & pl.col(EDGE_DROP_COLUMN).is_not_null()
        & (pl.col(EDGE_SIDE_COLUMN) == pl.col(EDGE_DROP_COLUMN))
    )

    return (
        annotated.join(edge_decisions, on=["qseqid", "sseqid"], how="left")
        .filter(~drop_mask)
        .drop([EDGE_SIDE_COLUMN, EDGE_DROP_COLUMN])
        .select(BLAST_COLUMNS)
    )


def _sort_frame_for_overlap(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(
        ["sseqid", "sstart", "send", "bitscore"],
        descending=[False, False, False, True],
    )


def _rows_overlap(
    previous_end: int, current_start: int, overlap_tolerance: int
) -> bool:
    if current_start > previous_end:
        return False
    return (previous_end - current_start) > overlap_tolerance


def _retain_non_overlapping_row_indexes(
    frame: pl.DataFrame, overlap_tolerance: int
) -> list[int]:
    if frame.is_empty():
        return []

    retained_positions: list[int] = []
    survivor_stack: list[tuple[int, int, float]] = []
    current_contig_id: str | None = None

    for (
        raw_row_index,
        raw_contig_id,
        raw_contig_start,
        raw_contig_end,
        raw_score,
    ) in frame.select(
        [ROW_INDEX_COLUMN, "sseqid", "sstart", "send", "bitscore"]
    ).iter_rows():
        row_index = int(raw_row_index)
        contig_id = str(raw_contig_id)
        contig_start = int(raw_contig_start)
        contig_end = int(raw_contig_end)
        score = float(raw_score)
        if current_contig_id != contig_id:
            retained_positions.extend(
                survivor_row_index for survivor_row_index, _, _ in survivor_stack
            )
            survivor_stack = [(row_index, contig_end, score)]
            current_contig_id = contig_id
            continue

        current_survives = True

        while survivor_stack:
            _previous_row_index, previous_end, previous_score = survivor_stack[-1]
            if not _rows_overlap(previous_end, contig_start, overlap_tolerance):
                break

            if score > previous_score:
                survivor_stack.pop()
                continue

            current_survives = False
            break

        if current_survives:
            survivor_stack.append((row_index, contig_end, score))

    retained_positions.extend(
        survivor_row_index for survivor_row_index, _, _ in survivor_stack
    )
    return retained_positions


def _filter_overlaps_frame(
    frame: pl.DataFrame, overlap_tolerance: int
) -> pl.DataFrame:
    if frame.is_empty():
        return frame.select(BLAST_COLUMNS)
    indexed = frame.with_row_index(ROW_INDEX_COLUMN)
    sorted_frame = _sort_frame_for_overlap(indexed)
    retained_row_indexes = _retain_non_overlapping_row_indexes(
        sorted_frame, overlap_tolerance
    )
    return (
        _sort_frame_for_overlap(
            indexed.filter(pl.col(ROW_INDEX_COLUMN).is_in(retained_row_indexes))
        )
        .drop(ROW_INDEX_COLUMN)
        .select(BLAST_COLUMNS)
    )


def _typing_thresholds(
    thresholds: Thresholds, biomarker_class: str
) -> _FrameThresholds:
    if biomarker_class == "replicon":
        return _FrameThresholds(
            min_length=thresholds.min_rep_length,
            min_cov=thresholds.min_rep_cov,
            min_hsp_cov=thresholds.min_rep_hsp_cov,
            min_ident=thresholds.min_rep_ident,
            evalue=thresholds.min_rep_evalue,
            max_query_length=thresholds.max_blast_query_length,
        )
    if biomarker_class == "relaxase":
        return _FrameThresholds(
            min_length=1,
            min_cov=thresholds.min_mob_cov,
            min_hsp_cov=thresholds.min_mob_hsp_cov,
            min_ident=thresholds.min_mob_ident,
            evalue=thresholds.min_mob_evalue,
        )
    if biomarker_class == "mate-pair-formation":
        return _FrameThresholds(
            min_length=1,
            min_cov=thresholds.min_mob_cov,
            min_hsp_cov=thresholds.min_mob_hsp_cov,
            min_ident=thresholds.min_mob_ident,
            evalue=thresholds.min_mob_evalue,
        )
    if biomarker_class == "oriT":
        return _FrameThresholds(
            min_length=thresholds.min_orit_length,
            min_cov=thresholds.min_rep_cov,
            min_hsp_cov=thresholds.min_orit_hsp_cov,
            min_ident=thresholds.min_rep_ident,
            evalue=thresholds.min_rep_evalue,
            max_query_length=thresholds.max_blast_query_length,
        )
    raise ValueError(f"Unknown biomarker class: {biomarker_class}")


def _report_thresholds(
    thresholds: Thresholds, biomarker_class: str
) -> _FrameThresholds:
    if biomarker_class == "replicon":
        return _FrameThresholds(
            min_length=80,
            min_cov=thresholds.min_rep_cov,
            min_hsp_cov=25.0,
            min_ident=thresholds.min_rep_ident,
            evalue=thresholds.min_rep_evalue,
            max_query_length=thresholds.max_blast_query_length,
        )
    if biomarker_class == "relaxase":
        return _FrameThresholds(
            min_length=40,
            min_cov=thresholds.min_mob_cov,
            min_hsp_cov=25.0,
            min_ident=thresholds.min_mob_ident,
            evalue=thresholds.min_mob_evalue,
        )
    if biomarker_class == "mate-pair-formation":
        return _FrameThresholds(
            min_length=40,
            min_cov=thresholds.min_mob_cov,
            min_hsp_cov=25.0,
            min_ident=thresholds.min_mob_ident,
            evalue=thresholds.min_mob_evalue,
        )
    if biomarker_class == "oriT":
        return _FrameThresholds(
            min_length=80,
            min_cov=thresholds.min_rep_cov,
            min_hsp_cov=15.0,
            min_ident=thresholds.min_rep_ident,
            evalue=thresholds.min_rep_evalue,
            max_query_length=thresholds.max_blast_query_length,
        )
    raise ValueError(f"Unknown biomarker class: {biomarker_class}")


def _build_hits_query(
    frame: pl.LazyFrame,
    *,
    thresholds: _FrameThresholds,
    remove_split_edges: bool,
) -> pl.LazyFrame:
    query = cast("pl.LazyFrame", _apply_thresholds(frame, thresholds))
    if remove_split_edges:
        query = _remove_split_edge_hits(query)
    return query.select(BLAST_COLUMNS)


def _build_biomarker_queries(
    raw_frames: RawBiomarkerFrames,
    thresholds: Thresholds,
    *,
    report: bool,
) -> RawBiomarkerFrames:
    threshold_factory = _report_thresholds if report else _typing_thresholds
    return RawBiomarkerFrames(
        replicon=_build_hits_query(
            raw_frames.replicon,
            thresholds=threshold_factory(thresholds, "replicon"),
            remove_split_edges=True,
        ),
        relaxase=_build_hits_query(
            raw_frames.relaxase,
            thresholds=threshold_factory(thresholds, "relaxase"),
            remove_split_edges=True,
        ),
        mate_pair_formation=_build_hits_query(
            raw_frames.mate_pair_formation,
            thresholds=threshold_factory(thresholds, "mate-pair-formation"),
            remove_split_edges=True,
        ),
        orit=_build_hits_query(
            raw_frames.orit,
            thresholds=threshold_factory(thresholds, "oriT"),
            remove_split_edges=False,
        ),
    )


def _prepare_biomarker_frames_from_queries(
    queries: RawBiomarkerFrames,
    *,
    overlap_tolerance: int,
) -> PreparedBiomarkerFrames:
    collected_frames = cast(
        "list[pl.DataFrame]",
        pl.collect_all(
            [
                queries.replicon,
                queries.relaxase,
                queries.mate_pair_formation,
                queries.orit,
            ]
        ),
    )
    replicon_frame, relaxase_frame, mate_pair_formation_frame, orit_frame = cast(
        "tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]",
        tuple(collected_frames),
    )
    return PreparedBiomarkerFrames(
        replicon=_filter_overlaps_frame(
            replicon_frame.select(BLAST_COLUMNS), overlap_tolerance
        ),
        relaxase=_filter_overlaps_frame(
            relaxase_frame.select(BLAST_COLUMNS), overlap_tolerance
        ),
        mate_pair_formation=_filter_overlaps_frame(
            mate_pair_formation_frame.select(BLAST_COLUMNS), overlap_tolerance
        ),
        orit=_filter_overlaps_frame(orit_frame.select(BLAST_COLUMNS), overlap_tolerance),
    )


def read_biomarker_frames(
    *,
    replicon_output: Path,
    relaxase_output: Path,
    mpf_output: Path,
    orit_output: Path,
) -> RawBiomarkerFrames:
    return RawBiomarkerFrames(
        replicon=scan_blast_table(replicon_output),
        relaxase=scan_blast_table(relaxase_output),
        mate_pair_formation=scan_blast_table(mpf_output),
        orit=scan_blast_table(orit_output),
    )


def prepare_typing_frames(
    raw_frames: RawBiomarkerFrames,
    thresholds: Thresholds,
) -> PreparedBiomarkerFrames:
    return _prepare_biomarker_frames_from_queries(
        _build_biomarker_queries(raw_frames, thresholds, report=False),
        overlap_tolerance=thresholds.overlap_tolerance,
    )


def prepare_report_frames(
    raw_frames: RawBiomarkerFrames,
    thresholds: Thresholds,
) -> PreparedBiomarkerFrames:
    return _prepare_biomarker_frames_from_queries(
        _build_biomarker_queries(raw_frames, thresholds, report=True),
        overlap_tolerance=thresholds.overlap_tolerance,
    )


def _records_frame(records: Sequence[RecordMetadata]) -> pl.DataFrame:
    return (
        pl.DataFrame(
            [
                {
                    "internal_id": record.accession,
                    "sample_id": record.accession,
                    "num_contigs": 1,
                    "size": record.size,
                    "gc": record.gc,
                    "md5": record.sequence_md5,
                }
                for record in records
            ],
            schema={
                "internal_id": pl.Utf8,
                "sample_id": pl.Utf8,
                "num_contigs": pl.Int64,
                "size": pl.Int64,
                "gc": pl.Float64,
                "md5": pl.Utf8,
            },
        )
        .with_row_index(RECORD_ORDER_COLUMN)
        .select(
            [
                RECORD_ORDER_COLUMN,
                "internal_id",
                "sample_id",
                "num_contigs",
                "size",
                "gc",
                "md5",
            ]
        )
    )


def _annotate_biomarker_hits(frame: pl.LazyFrame) -> pl.LazyFrame:
    identifier_parts = pl.col("qseqid").str.split_exact("|", 1)
    return (
        frame.with_row_index(HIT_ORDER_COLUMN)
        .with_columns(
            [
                identifier_parts.struct.field("field_0").alias("accession"),
                identifier_parts.struct.field("field_1").alias("biomarker_type"),
            ]
        )
        .select(
            [
                pl.col("sseqid").alias("internal_id"),
                "accession",
                "biomarker_type",
                HIT_ORDER_COLUMN,
            ]
        )
    )


def _unique_biomarker_hits(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _annotate_biomarker_hits(frame).unique(
        subset=["internal_id", "accession"],
        keep="first",
        maintain_order=True,
    )


def _summarize_biomarker_hits(
    frame: pl.LazyFrame,
    *,
    type_column: str,
    accession_column: str,
) -> pl.LazyFrame:
    return (
        _unique_biomarker_hits(frame)
        .sort(["internal_id", "biomarker_type", HIT_ORDER_COLUMN])
        .group_by("internal_id", maintain_order=True)
        .agg(
            [
                pl.col("biomarker_type").alias(type_column),
                pl.col("accession").alias(accession_column),
            ]
        )
    )


def _summarize_biomarker_accessions(
    frame: pl.LazyFrame,
    *,
    accession_column: str,
) -> pl.LazyFrame:
    return (
        _unique_biomarker_hits(frame)
        .sort(["internal_id", "biomarker_type", HIT_ORDER_COLUMN])
        .group_by("internal_id", maintain_order=True)
        .agg(pl.col("accession").alias(accession_column))
    )


def _summarize_mpf_type(frame: pl.LazyFrame) -> pl.LazyFrame:
    return (
        _unique_biomarker_hits(frame)
        .group_by(["internal_id", "biomarker_type"])
        .agg(pl.len().alias("type_count"))
        .sort(
            ["internal_id", "type_count", "biomarker_type"],
            descending=[False, True, False],
        )
        .unique(subset=["internal_id"], keep="first", maintain_order=True)
        .select(["internal_id", pl.col("biomarker_type").alias("mpf_type")])
    )


def _list_length(column_name: str) -> pl.Expr:
    return pl.col(column_name).list.len().fill_null(0)


def _predict_mobility_expr() -> pl.Expr:
    relaxase_present = _list_length("relaxase_types") > 0
    orit_present = _list_length("orit_types") > 0
    mpf_present = pl.col("mpf_type").fill_null("") != ""
    return (
        pl.when(relaxase_present & mpf_present)
        .then(pl.lit("conjugative"))
        .when(relaxase_present | orit_present)
        .then(pl.lit("mobilizable"))
        .otherwise(pl.lit("non-mobilizable"))
        .alias("predicted_mobility")
    )


def classify_records_query(
    records: Sequence[RecordMetadata],
    *,
    prepared_frames: PreparedBiomarkerFrames,
) -> pl.LazyFrame:
    records_frame = _records_frame(records)
    return (
        records_frame.lazy()
        .join(
            _summarize_biomarker_hits(
                prepared_frames.replicon.lazy(),
                type_column="rep_types",
                accession_column="rep_accessions",
            ),
            on="internal_id",
            how="left",
        )
        .join(
            _summarize_biomarker_hits(
                prepared_frames.relaxase.lazy(),
                type_column="relaxase_types",
                accession_column="relaxase_accessions",
            ),
            on="internal_id",
            how="left",
        )
        .join(
            _summarize_biomarker_accessions(
                prepared_frames.mate_pair_formation.lazy(),
                accession_column="mpf_accessions",
            ),
            on="internal_id",
            how="left",
        )
        .join(
            _summarize_mpf_type(prepared_frames.mate_pair_formation.lazy()),
            on="internal_id",
            how="left",
        )
        .join(
            _summarize_biomarker_hits(
                prepared_frames.orit.lazy(),
                type_column="orit_types",
                accession_column="orit_accessions",
            ),
            on="internal_id",
            how="left",
        )
        .with_columns(
            [
                pl.col("mpf_type").fill_null("").alias("mpf_type"),
                _predict_mobility_expr(),
            ]
        )
        .sort(RECORD_ORDER_COLUMN)
        .drop(["internal_id", RECORD_ORDER_COLUMN])
        .select(list(CLASSIFICATION_SCHEMA))
    )


def _list_to_output_string(column_name: str, alias: str) -> pl.Expr:
    return (
        pl.when(_list_length(column_name) > 0)
        .then(pl.col(column_name).list.join(","))
        .otherwise(pl.lit("-"))
        .alias(alias)
    )


def _formatted_gc_expr() -> pl.Expr:
    scaled_gc = (pl.col("gc").round(4) * 10_000).round(0).cast(pl.Int64)
    return pl.format(
        "{}.{}",
        (scaled_gc // 10_000).cast(pl.Utf8),
        (scaled_gc % 10_000).cast(pl.Utf8).str.zfill(4),
    ).alias("gc")


def format_classification_output_query(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.select(
        [
            "sample_id",
            "num_contigs",
            "size",
            _formatted_gc_expr(),
            "md5",
            _list_to_output_string("rep_types", "rep_type"),
            _list_to_output_string("rep_accessions", "rep_type_accession"),
            _list_to_output_string("relaxase_types", "relaxase_type"),
            _list_to_output_string(
                "relaxase_accessions", "relaxase_type_accession"
            ),
            pl.when(pl.col("mpf_type") != "")
            .then(pl.col("mpf_type"))
            .otherwise(pl.lit("-"))
            .alias("mpf_type"),
            _list_to_output_string("mpf_accessions", "mpf_type_accession"),
            _list_to_output_string("orit_types", "orit_type"),
            _list_to_output_string("orit_accessions", "orit_accession"),
            "predicted_mobility",
        ]
    )


def create_biomarker_report_query(
    *,
    prepared_report_frames: PreparedBiomarkerFrames,
) -> pl.LazyFrame:
    frames: list[pl.LazyFrame] = []
    for biomarker_class, attribute_name in BIOMARKER_SPECS:
        prepared = getattr(prepared_report_frames, attribute_name)
        if prepared.is_empty():
            continue
        mapped = prepared.lazy().with_columns(
            [
                pl.lit(biomarker_class).alias("biomarker"),
            ]
        )
        frames.append(mapped.select([*BLAST_COLUMNS, "biomarker"]))

    if not frames:
        return _empty_biomarker_report_query()
    return pl.concat(frames).select([*BLAST_COLUMNS, "biomarker"])
