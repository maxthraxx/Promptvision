import os
from PIL import Image
import pickle
from sdparsers import ParserManager
import hashlib
from pathlib import Path
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import streamlit as st
import pandas as pd


def get_hash(image):
    hash_digest = hashlib.sha256()
    with open(image, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_digest.update(chunk)

    return hash_digest.hexdigest()


class ImageRewardEngine:
    def __init__(self) -> None:
        import ImageReward as RM

        self.model = RM.load("ImageReward-v1.0")

    def score(self, positive_prompt, image):
        import torch
        with torch.no_grad():
            score = self.model.score(positive_prompt, image)
        return round(score, 3)


def index_directory(directory, ire, parser_manager, existing_images, imagereward=False):
    images = {}
    for root, dirs, files in os.walk(directory, topdown=True):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_path = Path(root) / file
                imghash = get_hash(image_path)

                if imghash not in existing_images:
                    image = Image.open(image_path)
                    exif = parser_manager.parse(image)
                    try:
                        positive_prompt = exif.prompts[0][0].value
                    except AttributeError:
                        positive_prompt = "No positive prompt found"
                    try:
                        negative_prompt = exif.prompts[0][1].value
                    except AttributeError:
                        negative_prompt = "No negative prompt found"
                    try:
                        metadata = exif.metadata
                    except AttributeError:
                        metadata = "No metadata found"
                    if imagereward:
                        imgscore = ire.score(positive_prompt, image)
                    else:
                        imgscore = 0.0

                    images[imghash] = {
                        "filename": image_path,
                        "width": image.width,
                        "height": image.height,
                        "positive_prompt": positive_prompt,
                        "negative_prompt": negative_prompt,
                        "metadata": metadata,
                        "imghash": imghash,
                        "score": imgscore,
                        "favorite": False,
                        "rating": 0,
                    }
                    print(f"Saved image: {image_path}")
        for subdir in dirs:
            index_directory(
                Path(os.path.join(root, subdir)),
                ire,
                parser_manager,
                existing_images,
                imagereward,
            )

    return images


def get_cached_images(directory):
    index_file = os.path.join(directory, "pvision_cache.pkl")
    if not os.path.exists(index_file):
        return {}

    with open(index_file, "rb") as f:
        try:
            images = pickle.load(f)
        except EOFError:
            return {}

    return images


def cache_images(directory, images):
    with open(os.path.join(directory, "pvision_cache.pkl"), "wb") as f:
        pickle.dump(images, f)


def save_df_from_streamlit(directory, df):
    # print(f"df in to save: {df}")
    images = df.set_index("imghash", drop=False).to_dict(orient="index")
    # print(f"dict from df: {images}")
    with open(os.path.join(directory, "pvision_cache.pkl"), "wb") as f:
        pickle.dump(images, f)


def process_directory(directory=None, imagereward=None, cleanup=None):
    if imagereward:
        ire = ImageRewardEngine()
    else:
        ire = None

    parser_manager = ParserManager(process_items=True)

    if directory is None:
        directory = args.imagedir
    assert Path(directory).is_dir()
    print(f"Directory --> {directory}")

    if cleanup or imagereward:
        if os.path.exists(Path(os.path.join(directory, "pvision_cache.pkl"))):
            os.remove(Path(os.path.join(directory, "pvision_cache.pkl")))
        print("Cache cleared.")
        existing_images = []
    else:
        # Load the existing images from the cache first
        existing_images = get_cached_images(directory)
        print(f"existing_images: {existing_images}")
    # Index the directory and find the new images that are not in the cache
    images = index_directory(
        directory, ire, parser_manager, existing_images, imagereward
    )
    new_images = {
        imghash: image
        for imghash, image in images.items()
        if imghash not in existing_images
    }

    # Process the new images and update the cache
    if new_images:
        cache_images(directory, new_images)
        print(f"new_images: {new_images}")

    images.update(new_images)
    images.update(existing_images)
    print(f"Images before creating df: {images}")

    # Check if any of the values are NA or inf
    for imghash, image_info in images.items():
        for key, value in image_info.items():
            if pd.isna(value):
                print(f"Invalid value {value} for {key} in image {imghash}")
    # Create the dataframe from the images dictionary
    df = pd.DataFrame.from_dict(images, orient="index").reset_index(drop=True)
    if not df.empty:
        # Set the datatypes for the columns
        try:
            df["filename"] = df["filename"].astype("str")
            df["width"] = df["width"].astype("int64")
            df["height"] = df["height"].astype("int64")
            df["positive_prompt"] = df["positive_prompt"].astype("str")
            df["negative_prompt"] = df["negative_prompt"].astype("str")
            df["metadata"] = df["metadata"].astype("str")
            df["imghash"] = df["imghash"].astype("str")
            df["score"] = df["score"].astype("float64")
            df["favorite"] = df["favorite"].astype("bool")
            df["rating"] = df["rating"].astype("int64")
            # print(df)
            # print(df.describe())
            # print(df.columns.to_list())
            # print(df.dtypes)
            return df
        except Exception as e:
            print(f"Error while applying astype: {e}")
    else:
        return pd.DataFrame()


def move_images(df, destination):
    for file in df["filename"].tolist():
        new_path = Path(destination) / Path(file).name
        Path(file).rename(new_path)


def copy_images(df, destination):
    for file in df["filename"].tolist():
        # Create a new path for the file in the directory
        new_path = Path(destination) / Path(file).name
        new_path.write_bytes(Path(file).read_bytes())


def delete_images(df):
    if df.empty:
        return
    for file in df["filename"].tolist():
        file_obj = Path(file)
        try:
            file_obj.unlink()
            print(f"Deleted: {file}")
        except FileNotFoundError:
            print(f"File not found: {file}")
        except PermissionError:
            print(f"Permission error: Cannot delete {file}")


def delete_images_and_cache(df):
    directory = set()
    for file in df["filename"].tolist():
        file_obj = Path(file)
        if file_obj.parent not in directory:
            directory.add(Path(os.path.join(file_obj.parent, "pvision_cache.pkl")))
        file_obj.unlink()
    for cache in directory:
        cache.unlink()


# Define the filter_dataframe function (modified from the blog post)
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")
    if not modify:
        return df

    df = df.copy()
    # Try to convert datetimes into a standard format (datetime, no timezone)
    to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
    for column in to_filter_columns:
        _, right = st.columns((1, 20))
        if is_numeric_dtype(df[column]):
            # Check if the score column has a constant value
            if df[column].nunique() == 1:
                # Use a text input instead of a slider
                user_num_input = right.text_input(
                    f"Value for {column}",
                    value=df[column].iloc[0],  # Use the constant value as default
                )
                df = df[df[column] == user_num_input]
            else:
                # Use a slider with a non-zero step
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = max((_max - _min) / 100, 0.01)  # Use a minimum step of 0.01
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
        elif is_datetime64_any_dtype(df[column]):
            user_date_input = right.date_input(
                f"Values for {column}",
                value=(
                    df[column].min(),
                    df[column].max(),
                ),
            )
            if len(user_date_input) == 2:
                user_date_input = tuple(map(pd.to_datetime, user_date_input))
                start_date, end_date = user_date_input
                df = df.loc[df[column].between(start_date, end_date)]
        else:
            # Use regex text search for these columns
            if column in ["positive_prompt", "negative_prompt", "metadata"]:
                user_text_input = right.text_input(
                    f"Regex in {column}",
                    value=".*",  # Default value to match all values
                )
                if user_text_input:
                    # Use regex=True and case=False to enable case-insensitive regex matching
                    df = df[
                        df[column]
                        .astype(str)
                        .str.contains(user_text_input, regex=True, case=False)
                    ]
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true", help="Clear the cache.")
    parser.add_argument("--copydir", action="store_true", help="Copy df to new dest.")
    parser.add_argument("--movedir", action="store_true", help="Move df to new dest.")
    parser.add_argument("--deletedir", action="store_true", help="Delete files in df.")
    parser.add_argument(
        "--imagereward",
        action="store_true",
        help="Enable ImageReward - https://github.com/THUDM/ImageReward.",
    )
    parser.add_argument("--imagedir", help="Path to your images", required=False)
    parser.add_argument(
        "--destinationdir", help="Path you want moving/copying to", required=False
    )

    args = parser.parse_args()
    if args.imagereward and args.cleanup:
        df = process_directory(directory=args.imagedir, imagereward=True, cleanup=True)
    elif args.imagereward:
        df = process_directory(directory=args.imagedir, imagereward=True, cleanup=False)
    elif args.cleanup:
        df = process_directory(directory=args.imagedir, imagereward=False, cleanup=True)
    else:
        df = process_directory(
            directory=args.imagedir, imagereward=False, cleanup=False
        )

    if args.copydir and args.destinationdir:
        copy_images(df, args.destinationdir)

    if args.movedir and args.destinationdir:
        move_images(df, args.destinationdir)

    if args.deletedir:
        delete_images(df)
