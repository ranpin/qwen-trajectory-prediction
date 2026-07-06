"""Download ETH/UCY and TrajNet++ trajectory prediction datasets."""

import os
import json
import csv
import zipfile
import urllib.request
import argparse
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Alternative source: Social-STGCNN (stable and accessible)
ETH_UCY_SCENES = {
    "eth": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/eth/train/biwi_eth.txt",
    "hotel": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/hotel/train/biwi_hotel.txt",
    "univ": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/univ/train/students001.txt",
    "zara1": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/zara1/train/crowds_zara01.txt",
    "zara2": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/zara2/train/crowds_zara02.txt",
}

ETH_UCY_TEST = {
    "eth": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/eth/test/biwi_eth.txt",
    "hotel": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/hotel/test/biwi_hotel.txt",
    "univ": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/univ/test/students003.txt",
    "zara1": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/zara1/test/crowds_zara01.txt",
    "zara2": "https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/main/datasets/raw/zara2/test/crowds_zara02.txt",
}


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  [download] {url.split('/')[-1]}")
    urllib.request.urlretrieve(url, str(dest))


def download_eth_ucy():
    """Download ETH/UCY dataset using Hugging Face datasets library."""
    out_dir = DATA_DIR / "eth_ucy"
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ETH/UCY using Hugging Face datasets...")
    try:
        from datasets import load_dataset

        # Load ETH/UCY from Hugging Face
        dataset = load_dataset("trajnetplusplus/trajnetplusplusdata", "eth")

        # Save train split
        if "train" in dataset:
            for scene in ["eth", "hotel", "univ", "zara1", "zara2"]:
                try:
                    scene_data = load_dataset("trajnetplusplus/trajnetplusplusdata", scene)
                    if "train" in scene_data:
                        train_file = train_dir / f"{scene}.txt"
                        with open(train_file, "w") as f:
                            for item in scene_data["train"]:
                                # Format: frame_id ped_id x y
                                for frame_id, ped_id, x, y in zip(
                                    item.get("frame", []),
                                    item.get("ped_id", []),
                                    item.get("x", []),
                                    item.get("y", [])
                                ):
                                    f.write(f"{frame_id}\t{ped_id}\t{x}\t{y}\n")
                        print(f"  Saved train/{scene}.txt")

                    if "test" in scene_data:
                        test_file = test_dir / f"{scene}.txt"
                        with open(test_file, "w") as f:
                            for item in scene_data["test"]:
                                for frame_id, ped_id, x, y in zip(
                                    item.get("frame", []),
                                    item.get("ped_id", []),
                                    item.get("x", []),
                                    item.get("y", [])
                                ):
                                    f.write(f"{frame_id}\t{ped_id}\t{x}\t{y}\n")
                        print(f"  Saved test/{scene}.txt")
                except Exception as e:
                    print(f"  Warning: Could not load {scene}: {e}")

    except ImportError:
        print("  Hugging Face datasets not installed. Install with: pip install datasets")
        print("  Alternatively, download manually from:")
        print("    https://github.com/vita-epfl/trajnetplusplusdata/releases")
        print(f"    and place files in: {out_dir}")
        return
    except Exception as e:
        print(f"  Error downloading dataset: {e}")
        print("  Please download manually or use synthetic data only.")
        return

    print(f"ETH/UCY saved to {out_dir}")


def download_trajnetpp():
    out_dir = DATA_DIR / "trajnetpp"
    out_dir.mkdir(parents=True, exist_ok=True)

    releases_url = "https://github.com/vita-epfl/trajnetplusplusdata/releases"
    print(f"TrajNet++ must be downloaded manually from:")
    print(f"  {releases_url}")
    print(f"Place the files in: {out_dir}")
    print(f"Alternatively, using TrajNet++ synthetic data from Trajectron++...")

    trajnet_synth = "https://raw.githubusercontent.com/lmb-freiburg/trajectron-plus-plus/master/experiments/pedestrians/raw/README.md"
    readme = out_dir / "README.md"
    download_file(trajnet_synth, readme)
    print(f"Placeholder saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download trajectory datasets")
    parser.add_argument("--datasets", nargs="+", default=["eth_ucy"],
                        choices=["eth_ucy", "trajnetpp"],
                        help="Which datasets to download")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if "eth_ucy" in args.datasets:
        download_eth_ucy()
    if "trajnetpp" in args.datasets:
        download_trajnetpp()

    print("\nDone. Dataset summary:")
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            files = list(d.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  {d.name}: {len(files)} files, {total_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
