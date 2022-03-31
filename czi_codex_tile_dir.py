from typing import Union, Optional
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import czifile
import numpy as np
from tifffile import imwrite
from tqdm import tqdm


def read_and_write_tile(
    sub_block_directory: czifile.DirectoryEntryDV,
) -> None:
    """
    mapped function to read tile metadata and data and write to disk

    Parameters
    ----------
    sub_block_directory: czifile.DirectoryEntryDV
        czi data sub-block

    """
    tile_name_template = "Cyc{cycle:03d}_reg{region:03d}/{region:d}_{tile:05d}_Z{zplane:03d}_CH{channel:d}.tif"

    # get c and z indices
    c_axis_idx = sub_block_directory.axes.index("C")
    z_axis_idx = sub_block_directory.axes.index("Z")

    # get c and z axis position for data sub-block
    c_dim_entry = sub_block_directory.dimension_entries[c_axis_idx + 1]
    z_dim_entry = sub_block_directory.dimension_entries[z_axis_idx + 1]

    c_idx = c_dim_entry.start
    z_idx = z_dim_entry.start
    tile_idx = sub_block_directory.mosaic_index

    # read data into python
    data_seg = sub_block_directory.data_segment()
    im = data_seg.data()

    # construct name
    tile_out_name = tile_name_template.format(
        cycle=cyc_idx_in,
        region=region_no_in,
        tile=tile_idx + 1,
        zplane=z_idx + 1,
        channel=c_idx + 1,
    )
    output_path = Path(output_dir_in) / tile_out_name
    # write data
    imwrite(output_path, im, compression="deflate")


def convert_czi_to_tiles(
    submission_dir: Union[str, Path],
    max_workers: Optional[int] = None,
    test_tiles: Optional[int] = None,
) -> None:
    """
    Reads a CODEX zeiss .czi and dumps the tiles onto disk mimicing akoya layout

    Parameters
    ----------
    submission_dir : Union[str, Path]
        file path to submission directory containing raw, processed, extras directories
        Will recursively search for 'extras' directory
    max_workers: Optional[int]
        how many multiprocessing works to use. If none, it will detect number of
        cores and use all
    """

    extras_dir = sorted(Path(submission_dir).rglob("extras/"))

    if len(extras_dir) == 0:
        raise ValueError(f"Submission directory does not contain an extras directory")

    czi_files = sorted(extras_dir[0].glob("*.czi"))

    for idx, czi_fp in enumerate(czi_files):
        czi = czifile.CziFile(czi_fp)
        sbs = czi.filtered_subblock_directory

        tile_indices = [s.mosaic_index for s in sbs]
        tile_indices, tile_counts = np.unique(tile_indices, return_counts=True)

        if np.all(tile_counts) is False:
            raise ValueError("not all tiles contain the same number of c,z,y,x planes")

        # make sure these variables are in scope for the ThreadPoolExecutor
        # mapping an iterator with some other variables is a pain
        global cyc_idx_in
        global region_no_in
        global output_dir_in

        cyc_idx_in = idx + 1
        region_no_in = 1
        output_dir_in = Path(submission_dir) / "raw"

        out_dir = (
            Path(output_dir_in)
            / f"Cyc{str(cyc_idx_in).zfill(3)}_reg{str(region_no_in).zfill(3)}"
        )

        if not out_dir.exists():
            out_dir.mkdir(exist_ok=True)

        if not max_workers:
            max_workers = multiprocessing.cpu_count() - 1

        if test_tiles:
            test_tile_indices = np.sort(
                np.random.randint(0, len(czi.filtered_subblock_directory), test_tiles)
            )
            sbs = [czi.filtered_subblock_directory[i] for i in test_tile_indices]
        else:
            sbs = czi.filtered_subblock_directory

        if max_workers > 1:
            czi._fh.lock = True
            with ThreadPoolExecutor(max_workers) as executor:
                if test_tiles:
                    total_tiles = test_tiles
                else:
                    total_tiles = len(czi.filtered_subblock_directory)

                results = list(
                    tqdm(
                        executor.map(read_and_write_tile, sbs),
                        total=total_tiles,
                    )
                )
            czi._fh.lock = None
        else:
            for directory_entry in sbs:
                read_and_write_tile(directory_entry)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub_dir",
        type=Path,
        help="path to the HuBMAP submission "
        "directory containing extras, "
        "raw, and processed folder",
    )

    parser.add_argument(
        "--max_workers",
        nargs="?",
        help="number of workers to use in"
        " python multiprocessing. Default is to"
        " use max detected",
        const=None,
        type=int,
    )

    parser.add_argument(
        "--n_test_tiles",
        help="export a random subset of tiles to test process",
        nargs="?",
        const=None,
        type=int,
    )

    args = parser.parse_args()

    convert_czi_to_tiles(
        args.sub_dir, max_workers=args.max_workers, test_tiles=args.n_test_tiles
    )
