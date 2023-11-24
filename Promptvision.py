import streamlit as st
import pandas as pd
import pvision
from pathlib import Path
from streamlit.errors import StreamlitAPIException
import helper

st.set_page_config(
    page_title="Promptvision",
    page_icon="👀",
    layout="wide",
)

st.session_state.render_image = False

if "directory" not in st.session_state:
    st.session_state["directory"] = None

if "df" not in st.session_state:
    # Store the dataframe in the session state
    st.session_state["df"] = pd.DataFrame()

main_container = st.container()

# Check if index exists, if not then set to 0.
if "my_index" not in st.session_state:
    st.session_state.my_index = 0

if st.session_state["df"].empty:
    if st.session_state.directory is None:
        st.info("No directory is currently being browsed 🚨.")
    if "dir_warning" not in st.session_state:
        st.session_state.dir_warning = True

if not st.session_state["df"].empty and st.session_state.dir_warning:
    st.session_state.dir_warning = False

with main_container:
    if st.checkbox("Change main folder") or st.session_state.dir_warning:
        with st.form("dir_form"):
            st.session_state["directory"] = st.text_input("Enter a directory path", "")
            dir_submitted = st.form_submit_button("Set directory")
            imagereward_check = st.checkbox(
                "Calculate ImageReward score for each image?"
            )
            st.session_state.imagereward_check = imagereward_check
            if dir_submitted and imagereward_check:
                helper.set_directory(imagereward=st.session_state.imagereward_check)
            else:
                helper.set_directory(imagereward=st.session_state.imagereward_check)

    if st.checkbox("Filter dataset"):
        # Define the filter_dataframe function (modified from the blog post)
        if "original_df" not in st.session_state:
            st.session_state.original_df = st.session_state["df"].copy()
        else:
            st.session_state.original_df = st.session_state["df"].copy()

        st.session_state["df"] = pvision.filter_dataframe(st.session_state["df"])

    # Add a checkbox to enable directory filter
    filter_by_dir = st.checkbox("Change sub folder")
    if filter_by_dir:
        # Convert the filename column to Path objects
        st.session_state["df"]["filename"] = st.session_state["df"]["filename"].apply(
            Path
        )

        # Get the unique parent directories of the files
        directories = (
            st.session_state["df"]["filename"].apply(lambda x: x.parent).unique()
        )
        # Convert to list and append the current directory if not in the list
        directories = list(directories)
        if st.session_state.directory not in directories:
            directories += [st.session_state.directory]

        # Store the directories in a session state object
        if "directories" not in st.session_state:
            st.session_state.directories = directories

        # Add a multiselect widget to filter by directory
        selected_dir = st.selectbox("Filter by directory", st.session_state.directories)

        # Check if any directory is selected
        if selected_dir:
            # Filter the dataframe by the selected directories
            st.session_state["df"] = helper.load_data(
                selected_dir,
                cleanup=False,
                imagereward=st.session_state.imagereward_check,
            )
            st.session_state.my_index = int(st.session_state["df"].iloc[0].name)

        else:
            # Show a message when no directory is selected
            st.warning("Please select at least one directory")
            st.stop()

    if st.button("Reset cache"):
        helper.reset_cached_images()
    if st.button("Reset filter"):
        st.session_state["df"] = st.session_state.original_df.copy()

    st.write(f"Current directory: {st.session_state.directory}")

if st.session_state["df"].empty:
    st.warning("Enter a directory to view images, current view is empty.")
    st.stop()

if "original_df" in st.session_state:
    st.write(st.session_state.original_df)
