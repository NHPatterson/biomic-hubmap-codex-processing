import json
from copy import deepcopy
from typing import Union, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicspylibczi import CziFile
from tqdm import tqdm
from datetime import datetime

# data here is hardcoded to BIOMIC DR2 CODEX data
# collected on a Zeiss Axio Observer
MAIN_METADATA_TEMPLATE = {
    "Version": "1.0",
    "DatasetName": "Some recognizable name",
    "AcquisitionDate": "2020-02-19T13:51:35.857-05:00[America/New_York]",
    "AssayType": "CODEX",
    "AssaySpecificSoftware": "Zeiss Zen Blue (acquisition software)",
    "Microscope": "Zeiss AxioObserver",
    "AcquisitionMode": "WideField",
    "ImmersionMedium": "Air",
    "NominalMagnification": 20,
    "NumericalAperture": 0.8,
    "ResolutionX": 0.325,
    "ResolutionXUnit": "um",
    "ResolutionY": 0.325,
    "ResolutionYUnit": "um",
    "ResolutionZ": 1.5,
    "ResolutionZUnit": "um",
    "BitDepth": 16,
    "NumRegions": 1,
    "NumCycles": 4,
    "NumZPlanes": 5,
    "NumChannels": 6,
    "RegionWidth": 10,
    "RegionHeight": 10,
    "TileWidth": 2048,
    "TileHeight": 2048,
    "TileOverlapX": 0.3,
    "TileOverlapY": 0.3,
    "TileLayout": "Snake",
}

SEGMENTATION_META_TEMPLATE = {
    "NuclearStain": [{"CycleID": 2, "ChannelID": 1}],
    "MembraneStain": [{"CycleID": 2, "ChannelID": 3}, {"CycleID": 3, "ChannelID": 4}],
    "NuclearStainForSegmentation": {"CycleID": 1, "ChannelID": 1},
    "MembraneStainForSegmentation": {"CycleID": 3, "ChannelID": 4},
}

CHANNEL_DETAILS_TEMPLATE = {
    "Name": "DAPI-01",
    "ChannelID": 1,
    "CycleID": 1,
    "Fluorophore": "DAPI",
    "PassedQC": True,
    "QCDetails": "Unspecific",
    "ExposureTimeMS": 10.0,
    "ExcitationWavelengthNM": 350,
    "EmissionWavelengthNM": 450,
    "Binning": 1,
    "Gain": 1.0,
}

CHANNEL_ID_TO_FLUOROPHORE = {1: "Hoechst 33342", 2: "FITC", 3: "DsRed", 4: "Cy5"}

FLUOROPHORE_EXCITATION = {"Hoechst 33342": 352, "FITC": 490, "DsRed": 558, "Cy5": 649}

FLUOROPHORE_EMISSION = {"Hoechst 33342": 461, "FITC": 490, "DsRed": 583, "Cy5": 666}

# fixed for all BIOMIC CODEX in DR2
CHANNEL_ID_TO_EXPOSURE_TIMES = {1: 10, 2: 250, 3: 350, 4: 350}


def _read_channel_names_and_QC(
    channel_report_csv: Union[str, Path]
) -> Tuple[List[str], List[bool]]:
    """
    Read channel reports .csv
    """
    channels_report = pd.read_csv(channel_report_csv)

    channel_names = channels_report["Marker"].tolist()
    channel_passed_qc = channels_report["Result"].tolist()

    return channel_names, channel_passed_qc


class NpEncoder(json.JSONEncoder):
    """Ensure correct encoding of numpy data types
    in json file"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_hubmap_codex_metadata(
    submission_dir: Union[str, Path],
    channel_report_csv: Union[str, Path],
    nuc_stain_cycle: int = 2,
    membrane_stain_cycle: int = 2,
    membrane_stain_channel: int = 2,
    name: Optional[str] = None,
) -> str:
    """

    Parameters
    ----------
    submission_dir: path
        path to VU BIOMIC HuBMAP CODEX submission directory
    channel_report_csv: path
        path to VU BIOMIC channel reports .csv that provides channel name and
        whether the channel passed QC in a two column .csv with headers
        "Marker", "Result"
    nuc_stain_cycle: int
        Which CODEX cycle's nuclei staining should be used for segmentation
    membrane_stain_cycle: int
        Which CODEX cycle contains the membrane marker used for segmentation
    membrane_stain_channel: int
        Within cycle selected by `membrane_stain_cycle`, which channel should be used
        Channels are indexed starting from 1
    name: str
        A name for the dataset, otherwise inferred from file paths

    Returns
    -------
    metadata_out_fp: str
        Path to the output metadata experiment.json
    """

    # start metadata from template
    full_metadata = deepcopy(MAIN_METADATA_TEMPLATE)
    segm_metdata = deepcopy(SEGMENTATION_META_TEMPLATE)

    czi_data_dir = sorted(Path(submission_dir).rglob("extras/"))
    czi_files = sorted(Path(czi_data_dir).glob("*.czi"))
    n_cycle = len(czi_files)
    czi_fp = str(czi_files[0])

    # get channel metadata
    channel_names, channel_passed_qc = _read_channel_names_and_QC(channel_report_csv)

    # initialize readers
    czi = CziFile(czi_fp)
    czi_aics = AICSImage(czi_fp, reconstruct_mosaic=False)

    # in micron
    x_spacing = czi_aics.ome_metadata.images[0].pixels.physical_size_x
    y_spacing = czi_aics.ome_metadata.images[0].pixels.physical_size_y
    z_spacing = czi_aics.ome_metadata.images[0].pixels.physical_size_z

    acq_data = czi_aics.ome_metadata.images[0].acquisition_date

    # n tiles
    max_m = czi.get_dims_shape()[0]["M"][1]

    # figure out the tile grid
    x_pos, y_pos = [], []
    x_tile_size, y_tile_size = [], []

    print("reading tile metadata")

    for m in tqdm(range(max_m)):
        metadata = czi.read_subblock_metadata(C=0, M=m, Z=0, unified_xml=True)
        stage_x_pos = float(
            metadata.findall(".//StageXPosition")[0]
            .text.replace("+", "")
            .replace("-", "")
        )
        stage_y_pos = float(
            metadata.findall(".//StageYPosition")[0]
            .text.replace("+", "")
            .replace("-", "")
        )

        _, _, x_size, y_size = [
            int(f) for f in metadata.findall(".//Frame")[0].text.split(",")
        ]

        x_pos.append(stage_x_pos)
        y_pos.append(stage_y_pos)
        x_tile_size.append(x_size)
        y_tile_size.append(y_size)

    df = pd.DataFrame(
        {
            "instrument_stage_x_um": x_pos,
            "instrument_stage_y_um": y_pos,
            "tile_size_x_px": x_size,
            "tile_size_y_px": y_size,
        }
    )
    df["stage_x_minimized_um"] = df["instrument_stage_x_um"] - np.min(
        df["instrument_stage_x_um"]
    )
    df["stage_y_minimized_um"] = df["instrument_stage_y_um"] - np.min(
        df["instrument_stage_y_um"]
    )

    df["stage_x_px"] = df["stage_x_minimized_um"] / x_spacing
    df["stage_y_px"] = df["stage_y_minimized_um"] / y_spacing

    # get loose grid of tile positions
    df["grid_x"] = np.round(np.around(df["stage_x_minimized_um"], -1) / 600) * 1200
    df["grid_y"] = np.round(np.around(df["stage_y_minimized_um"], -1) / 600) * 1200

    if not name:
        full_metadata["DatasetName"] = Path(czi_fp).name.split("-cyc")[0]
    else:
        full_metadata["DatasetName"] = name

    full_metadata["AcquisitionDate"] = acq_data.strftime("%Y-%m-%dT%H:%M:%S")
    full_metadata["NumZPlanes"] = czi_aics.dask_data.shape[3]

    full_metadata["RegionWidth"] = len(df["grid_x"].unique())
    full_metadata["RegionHeight"] = len(df["grid_y"].unique())
    full_metadata["NumCycles"] = n_cycle
    full_metadata["NumChannels"] = czi_aics.dask_data.shape[2] * n_cycle

    full_metadata["TileHeight"] = 2048
    full_metadata["TileWidth"] = 2048
    full_metadata["TileOverlapX"] = 0.1
    full_metadata["TileOverlapY"] = 0.1

    full_metadata["ResolutionX"] = x_spacing
    full_metadata["ResolutionY"] = y_spacing
    full_metadata["ResolutionZ"] = z_spacing

    segm_metdata["NuclearStain"] = [
        {"CycleID": i + 1, "ChannelID": 1} for i in range(n_cycle)
    ]
    segm_metdata["MembraneStain"] = [
        {"CycleID": membrane_stain_cycle, "ChannelID": membrane_stain_channel}
    ]

    # nuclear stain is always channel 1 in BIOMIC CODEX
    segm_metdata["NuclearStainForSegmentation"] = {
        "CycleID": nuc_stain_cycle,
        "ChannelID": 1,
    }
    segm_metdata["MembraneStainForSegmentation"] = {
        "CycleID": membrane_stain_cycle,
        "ChannelID": membrane_stain_channel,
    }

    channel_id_cycle = np.tile(np.arange(1, 5, 1), n_cycle)

    all_channel_details = []
    for idx, (channel_name, channel_pass) in enumerate(
        zip(channel_names, channel_passed_qc)
    ):
        cyc_id = int(np.ceil((idx + 1) / 4))
        current_cycle_channel_idx = channel_id_cycle[idx]
        fluorophore_id = CHANNEL_ID_TO_FLUOROPHORE[current_cycle_channel_idx]

        channel_meta = deepcopy(CHANNEL_DETAILS_TEMPLATE)
        channel_meta["Name"] = channel_name
        channel_meta["CycleID"] = cyc_id
        channel_meta["ChannelID"] = current_cycle_channel_idx
        channel_meta["ExposureTimeMS"] = CHANNEL_ID_TO_EXPOSURE_TIMES[
            current_cycle_channel_idx
        ]
        channel_meta["Fluorophore"] = fluorophore_id
        channel_meta["ExcitationWavelengthNM"] = FLUOROPHORE_EXCITATION[fluorophore_id]
        channel_meta["EmissionWavelengthNM"] = FLUOROPHORE_EMISSION[fluorophore_id]
        channel_meta["PassedQC"] = channel_pass
        channel_meta["QCDetails"] = "Passed" if channel_pass else "Unspecific"
        all_channel_details.append(channel_meta)

    full_metadata.update(segm_metdata)
    full_metadata.update(
        {"ChannelDetails": {"ChannelDetailsArray": all_channel_details}}
    )

    metadata_out_fp = Path(submission_dir) / "raw" / f"experiment.json"

    with open(metadata_out_fp, "w") as f:
        json.dump(full_metadata, f, cls=NpEncoder, indent=1)

    return str(metadata_out_fp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub_dir",
        type=Path,
        help="path to the HuBMAP submission "
        "directory containing extras, "
        "raw, and processed folder",
        required=True,
    )

    parser.add_argument(
        "--ch_rep", type=Path, help="path to the channel reports csv", required=True
    )

    parser.add_argument(
        "--nuc_cyc",
        type=int,
        help="Which CODEX cycle's nuclei staining should be used for segmentation",
        required=True,
    )
    parser.add_argument(
        "--mem_cyc",
        type=int,
        help="Which CODEX cycle contains the membrane " "marker used for segmentation",
        required=True,
    )
    parser.add_argument(
        "--mem_ch",
        type=int,
        help="Within cycle selected by --mem_cyc, "
        "which channel should be used. "
        "Channels are indexed starting from 1.",
        required=True,
    )

    parser.add_argument(
        "--name", type=str, help="Optional name for dataset", required=False
    )

    args = parser.parse_args()

    generate_hubmap_codex_metadata(
        args.sub_dir,
        args.ch_rep,
        nuc_stain_cycle=args.nuc_cyc,
        membrane_stain_cycle=args.mem_cyc,
        membrane_stain_channel=args.mem_ch,
        name=args.name,
    )
