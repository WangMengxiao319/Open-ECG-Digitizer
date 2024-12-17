import os
import random
import shutil
from tqdm import tqdm
from typing import List, Dict


def get_file_id(file: str) -> str:
    return file.split(".")[0].split("_")[0]


def generate_splits(src_timeseries_folder: str, src_masks_folder: str) -> Dict[str, List[str]]:
    """
    Val and test splits are randomly assigned from files in the timeseries folder.
    The train split consists of ids in the masks folder that are not in the val or test splits.
    """
    timeseries_ids = list(map(lambda x: x.split(".")[0], os.listdir(src_timeseries_folder)))
    random.shuffle(timeseries_ids)
    val_ids = timeseries_ids[: len(timeseries_ids) // 2]
    test_ids = timeseries_ids[len(timeseries_ids) // 2 :]

    train_ids = list(
        filter(lambda x: x and x not in val_ids + test_ids, map(get_file_id, os.listdir(src_masks_folder)))
    )

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def copy_data(folder_pairs: Dict[str, str], splits: Dict[str, List[str]]) -> None:
    for src_folder, dst_folder in folder_pairs.items():
        for split_name, split_ids in splits.items():
            if not os.path.exists(os.path.join(dst_folder, split_name)):
                os.makedirs(os.path.join(dst_folder, split_name))

            for id in tqdm(split_ids, desc=f"Copying {src_folder} to {dst_folder}/{split_name}"):
                files = filter(lambda f: get_file_id(f) == id, os.listdir(src_folder))
                for file in files:
                    shutil.copyfile(os.path.join(src_folder, file), os.path.join(dst_folder, split_name, file))

    for dst_folder in folder_pairs.values():
        for split_name in splits.keys():
            folder_location = os.path.join(dst_folder, split_name)
            print(f"{folder_location} size: {len(os.listdir(folder_location))}")

    return None


if __name__ == "__main__":
    random.seed(42)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    DELETE_PREVIOUS_DST_FOLDERS = False
    if not DELETE_PREVIOUS_DST_FOLDERS:

        print(
            """
Set DELETE_PREVIOUS_DST_FOLDERS to True when generating the dataset to ensure that no entries from previous runs are
 left in the destination folders. Defaulting to False to prevent accidental deletion of data.
            """
        )

    # Change the following paths to match the location of the ECG dataset on
    # your local machine. The paths should be relative to the location of this
    # file.
    SRC_TIMESERIES_FOLDER = "../../data/other/ecg_timeseries_1000"
    SRC_MASKS_FOLDER = "../../data/other/ecg_masks/masks"
    SRC_SCANS_FOLDER = "../../data/other/ecg_scans"
    DST_TIMESERIES_FOLDER = "../../data/ecg/ecg_timeseries_1000"
    DST_MASKS_FOLDER = "../../data/ecg/ecg_masks"
    DST_SCANS_FOLDER = "../../data/ecg/ecg_scans"

    folder_pairs = {
        SRC_TIMESERIES_FOLDER: DST_TIMESERIES_FOLDER,
        SRC_MASKS_FOLDER: DST_MASKS_FOLDER,
        SRC_SCANS_FOLDER: DST_SCANS_FOLDER,
    }

    for src_folder, dst_folder in folder_pairs.items():
        if not os.path.exists(src_folder):
            raise ValueError(f"{src_folder} does not exist")
        if DELETE_PREVIOUS_DST_FOLDERS and os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)

    splits = generate_splits(SRC_TIMESERIES_FOLDER, SRC_MASKS_FOLDER)
    copy_data(folder_pairs, splits)
    print("Finished generating ECG dataset.")
