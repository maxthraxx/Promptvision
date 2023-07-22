import streamlit as st
import base64
import pvision
from pathlib import Path


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


@st.cache_data
def load_data(directory, cleanup, imagereward=False):
    # Process the directory and create a dataframe of images and their metadata
    if directory == None:
        st.warning("Enter a directory to view")
        st.stop()
    else:
        df = pvision.process_directory(
            directory=directory, cleanup=cleanup, imagereward=imagereward
        )
    return df


def set_directory(cleanup=False, imagereward=False):
    st.session_state["df"] = load_data(
        st.session_state["directory"], cleanup=cleanup, imagereward=imagereward
    )


def reset_cached_images():
    st.cache_resource.clear()
    st.cache_data.clear()
    st.session_state["df"] = load_data(
        st.session_state["directory"],
        cleanup=True,
        imagereward=st.session_state.imagereward_check,
    )


def store_metadata_in_session(index):
    # Store the image and its metadata in the session state
    st.session_state["filename"] = st.session_state["df"].loc[index, "filename"]
    st.session_state["width"] = st.session_state["df"].loc[index, "width"]
    st.session_state["height"] = st.session_state["df"].loc[index, "height"]
    st.session_state["positive_prompt"] = st.session_state["df"].loc[
        index, "positive_prompt"
    ]
    st.session_state["negative_prompt"] = st.session_state["df"].loc[
        index, "negative_prompt"
    ]
    st.session_state["metadata"] = st.session_state["df"].loc[index, "metadata"]
    st.session_state["imghash"] = st.session_state["df"].loc[index, "imghash"]
    st.session_state["score"] = st.session_state["df"].loc[index, "score"]
    st.session_state["favorite"] = st.session_state["df"].loc[index, "favorite"]
    st.session_state["rating"] = st.session_state["df"].loc[index, "rating"]
