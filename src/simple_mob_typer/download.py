import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_MARKER_DB_URL = (
    "https://zenodo.org/records/10304948/files/data.tar.gz?download=1"
)
REQUIRED_MARKER_FILES = (
    "rep.dna.fas",
    "mob.proteins.faa",
    "mpf.proteins.faa",
    "orit.fas",
)


def download_archive(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(url) as response,
        destination.open("wb") as output_handle,
    ):
        shutil.copyfileobj(response, output_handle)
    return destination


def _member_target_name(member_name: str) -> str | None:
    cleaned_name = member_name.rstrip("/")
    filename = Path(cleaned_name).name
    if filename in REQUIRED_MARKER_FILES:
        return filename
    return None


def extract_required_marker_files(
    archive_path: Path, database_dir: Path, *, force: bool
) -> list[Path]:
    database_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    found_filenames: set[str] = set()

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            target_name = _member_target_name(member.name)
            if target_name is None or not member.isfile():
                continue

            destination = database_dir / target_name
            if destination.exists() and not force:
                raise FileExistsError(
                    f"Refusing to overwrite existing file without --force: {destination}"
                )

            extracted_member = tar.extractfile(member)
            if extracted_member is None:
                continue

            with extracted_member, destination.open("wb") as output_handle:
                shutil.copyfileobj(extracted_member, output_handle)

            extracted.append(destination)
            found_filenames.add(target_name)

    missing = sorted(set(REQUIRED_MARKER_FILES) - found_filenames)
    if missing:
        raise FileNotFoundError(
            "Archive did not contain the required marker database files: "
            + ", ".join(missing)
        )
    return sorted(extracted)


def write_manifest(
    database_dir: Path, *, source_url: str, archive_path: Path | None
) -> Path:
    manifest_path = database_dir / "simple_mob_typer_db_manifest.json"
    manifest = {
        "source_url": source_url,
        "archive_path": str(archive_path) if archive_path is not None else None,
        "files": list(REQUIRED_MARKER_FILES),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def initialize_marker_databases(
    database_dir: Path,
    *,
    url: str = DEFAULT_MARKER_DB_URL,
    archive_path: Path | None = None,
    force: bool = False,
    keep_archive: bool = False,
) -> list[Path]:
    database_dir = database_dir.resolve()
    database_dir.mkdir(parents=True, exist_ok=True)

    cleanup_archive = False
    if archive_path is None:
        download_target = database_dir / "simple_mob_typer_markers.tar.gz"
        archive_path = download_archive(url, download_target)
        cleanup_archive = not keep_archive
    else:
        archive_path = archive_path.resolve()

    extracted = extract_required_marker_files(archive_path, database_dir, force=force)
    write_manifest(
        database_dir,
        source_url=url,
        archive_path=archive_path if keep_archive else None,
    )

    if cleanup_archive and archive_path.exists():
        archive_path.unlink()

    return extracted
