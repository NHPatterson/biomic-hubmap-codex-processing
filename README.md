# Tools for VU BIOMIC's CODEX multiplex immunofluorescence data

This repository contains scripts to generate CODEX metadata to meet HuBMAP's standard and convert raw CODEX data
from Zeiss .czi files to HuBMAP's format for processing with the HuBMAP pipeline. 


## Converting Zeiss .czi files to HuBMAP tiles for processing

The `czi_codex_tile_dir.py` is a command line utility that takes a Vanderbilt TMC CODEX submission directory
and converts the tiles to .tiff in the probably format for this [HuBMAP pipeline](https://github.com/hubmapconsortium/codex-pipeline).

This utility will loop over all .czi files in a submission directory and write their tiles.

### Usage

#### Basic Usage
```bash
python czi_codex_tile_dir.py "path/to/submission/directory"
```

#### Testing a small sub-set of the data

This process by itself is long-running and will need >1 TB for conversion of the data. Using the '--n_tile_tiles' 
argument will randomly convert a small number of tiles for each cycle in a codex run to quickly determine if
it is working.

```bash
# get 10 tiles from each cycle
python czi_codex_tile_dir.py "path/to/submission/directory" --n_test_tiles 10
```

#### Changing the number of multiprocessing workers

In some cases, using many workers for multiprocessing may actually be slower. This can be changed with the `--max_workers` argument.
By default this will use `multiprocessing` to determine the maximum number available, but if `--max_workers 0` is used,
the conversion will not use multiprocessing.

```bash
# use 4 workers
python czi_codex_tile_dir.py "path/to/submission/directory" --max_workers 4

# don't use multiprocessing
python czi_codex_tile_dir.py "path/to/submission/directory" --max_workers 0
```


