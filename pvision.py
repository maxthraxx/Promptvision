import os
from PIL import Image, ExifTags
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
import simple_logger
import json
from concurrent.futures import ThreadPoolExecutor

# Create a logger object using the create_logger function
logger = simple_logger.create_logger("pvision_logger")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_PROMPT = "None"
DEFAULT_METADATA = "None"
PROMPT_FIELD = "1337"
POSITIVE_PROMPT_FIELD = "positive_prompt"
NEGATIVE_PROMPT_FIELD = "negative_prompt"


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
            logger.debug(positive_prompt)
            if positive_prompt:
                score = self.model.score(positive_prompt, image)
            else:
                score = 0.0
        
        return round(score, 3)


def process_image(image_path, imghash, ire, parser_manager, existing_images, imagereward):
    image = Image.open(image_path)
    logger.debug(image_path)

    exif = parser_manager.parse(image)
    logger.debug(exif)

    metadata, positive_prompt, negative_prompt = extract_metadata_and_prompts(exif, image)

    if imagereward:
        imgscore = ire.score(positive_prompt, image)
    else:
        imgscore = 0.0

    return imghash, {
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

def index_directory(directory, ire, parser_manager, existing_images, imagereward=False):
    images = {}
    image_tasks = []

    with ThreadPoolExecutor(max_workers=50) as executor:
        for root, dirs, files in os.walk(directory, topdown=True):
            for file in files:
                if file.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    image_path = os.path.join(root, file)
                    imghash = get_hash(image_path)
                    if imghash not in existing_images:
                        image_tasks.append(executor.submit(process_image, image_path, imghash, ire, parser_manager, existing_images, imagereward))

            for subdir in dirs:
                # Process subdirectories concurrently
                subdir_path = os.path.join(root, subdir)
                image_tasks.append(executor.submit(index_directory, subdir_path, ire, parser_manager, existing_images, imagereward))

    # Wait for all tasks to complete
    for future in image_tasks:
        try:
            result = future.result()
            if result:
                imghash, result_dict = result
                images[imghash] = result_dict
                logger.debug(f"Saved image: {result_dict['filename']}")
        except Exception as e:
            logger.error(f"Error processing image: {e}")

    return images


def extract_metadata_and_prompts(exif, image):
    metadata = DEFAULT_METADATA
    positive_prompt = DEFAULT_PROMPT
    negative_prompt = DEFAULT_PROMPT

    if exif:
        try:
            if "ComfyUI" == exif.generator:
                logger.debug(exif.raw_params["prompt"])
                logger.debug(exif.raw_params["workflow"])
                metadata = exif.raw_params
                parsed_prompt = json.loads(metadata["prompt"])
                positive_prompt = parsed_prompt.get(PROMPT_FIELD, {}).get(POSITIVE_PROMPT_FIELD)
                negative_prompt = parsed_prompt.get(PROMPT_FIELD, {}).get(NEGATIVE_PROMPT_FIELD)
            else:
                metadata = exif.metadata
                for prompt, n_prompt in exif.prompts:
                    if prompt:
                        positive_prompt = prompt.value
                    else:
                        positive_prompt = "No positive prompt found"
                    if n_prompt:
                        negative_prompt = n_prompt.value
                    else:
                        negative_prompt = "No negative prompt found"
            logger.debug(metadata)
        except AttributeError:
            metadata = {"metadata": "No metadata found"}
    else:
        # Read exif data custom (tested on ComfyUI API generated images)
        img_exif = image.info
        logger.debug(img_exif)
        if img_exif is None:
            logger.debug("Sorry, image has no exif data.")
        else:
            metadata = img_exif
            # Accessing the "prompt" field and extracting the "positive_prompt" and "negative_prompt"
            try:
                parsed_prompt = json.loads(metadata["prompt"])
            except KeyError as e:
                logger.error(e)
                parsed_prompt = {}
            positive_prompt = parsed_prompt.get(PROMPT_FIELD, {}).get(POSITIVE_PROMPT_FIELD)
            negative_prompt = parsed_prompt.get(PROMPT_FIELD, {}).get(NEGATIVE_PROMPT_FIELD)

    return metadata, positive_prompt, negative_prompt


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
    # logger.debug(f"df in to save: {df}")
    images = df.set_index("imghash", drop=False).to_dict(orient="index")
    # logger.debug(f"dict from df: {images}")
    with open(os.path.join(directory, "pvision_cache.pkl"), "wb") as f:
        pickle.dump(images, f)


def process_directory(directory=None, imagereward=None, cleanup=None):
    ire = ImageRewardEngine() if imagereward else None
    parser_manager = ParserManager(process_items=True)

    # Handle directory
    assert Path(directory).is_dir(), f"Invalid directory: {directory}"
    logger.debug(f"Directory --> {directory}")

    # Handle cache
    if cleanup or imagereward:
        cache_path = Path(directory) / "pvision_cache.pkl"
        if cache_path.exists():
            os.remove(cache_path)
            logger.debug("Cache cleared.")
        existing_images = []
    else:
        existing_images = get_cached_images(directory)
        logger.debug(f"Existing images: {len(existing_images)}")

    # Index and process new images
    images = index_and_process(directory, ire, parser_manager, existing_images, imagereward)

    # Update cache
    if images:
        cache_images(directory, images)
        logger.debug(f"New images: {images}")

    # Create DataFrame
    try:
        df = create_dataframe(images)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        return pd.DataFrame()

def index_and_process(directory, ire, parser_manager, existing_images, imagereward):
    images = index_directory(directory, ire, parser_manager, existing_images, imagereward)
    new_images = {imghash: image for imghash, image in images.items() if imghash not in existing_images}
    images.update(new_images)
    images.update(existing_images)
    logger.debug(f"Images before creating df: {len(images)}")
    return images

def create_dataframe(images):
    df = pd.DataFrame.from_dict(images, orient="index").reset_index(drop=True)
    if not df.empty:
        df = preprocess_dataframe(df)
    return df

def preprocess_dataframe(df):
    df["filename"] = df["filename"].astype("str")
    df["width"] = df["width"].astype("int64")
    df["height"] = df["height"].astype("int64")
    df["positive_prompt"] = df["positive_prompt"].astype("str").replace('None', '')
    df["negative_prompt"] = df["negative_prompt"].astype("str").replace('None', '')
    df["metadata"] = df["metadata"].astype("str")
    df["imghash"] = df["imghash"].astype("str")
    df["score"] = df["score"].astype("float64")
    df["favorite"] = df["favorite"].astype("bool")
    df["rating"] = df["rating"].astype("int64")
    return df

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
            logger.debug(f"Deleted: {file}")
        except FileNotFoundError:
            logger.debug(f"File not found: {file}")
        except PermissionError:
            logger.debug(f"Permission error: Cannot delete {file}")


def delete_images_and_cache(df):
    directory = set()
    for file in df["filename"].tolist():
        file_obj = Path(file)
        if file_obj.parent not in directory:
            directory.add(Path(os.path.join(file_obj.parent, "pvision_cache.pkl")))
        file_obj.unlink()
    for cache in directory:
        cache.unlink()


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
    to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
    
    for column in to_filter_columns:
        _, right = st.columns((1, 20))
        
        if is_numeric_dtype(df[column]):
            if df[column].nunique() == 1:
                # Use a text input instead of a slider for constant values
                user_num_input = right.text_input(
                    f"Value for {column}",
                    value=df[column].iloc[0],  # Use the constant value as default
                )
                # Convert user input to the numeric type of the column
                user_num_input = pd.to_numeric(user_num_input, errors="coerce")
                
                # Filter only if the user input is a valid number
                if not pd.isnull(user_num_input):
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
