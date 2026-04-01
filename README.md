# simple-mob-typer

`simple-mob-typer` is a tool for [MOB-typer](https://github.com/phac-nml/mob-suite?tab=readme-ov-file#mob-typer)-compatible plasmid typing. If you use this tool, make sure to cite [MOB-suite](https://github.com/phac-nml/mob-suite?tab=readme-ov-file#citations).

## Usage

You can execute `simple-mob-typer` using [Pixi](https://pixi.sh/):

```sh
pixi run simple-mob-typer --help
```

First, download the reference data:

```sh
pixi run simple-mob-typer download \
  --database-dir mob-suite-db
```

Process a FASTA file where each record represents a single plasmid genome:

```sh
pixi run simple-mob-typer run \
  --infile plasmids.fna \
  --out-file plasmids_simple_mob_typer.tsv \
  --database-dir mob-suite-db
```
