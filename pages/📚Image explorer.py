import streamlit as st
import pandas as pd
import pvision
from streamlit_image_select import image_select
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import helper

st.session_state.render_image = False

def filter_view():
    st.subheader("Drill down into your currently active images")
    st.write("This page lets you filter your current images without changing the the view such as filtering in the 'Promptvision' tab. When filtering here you can move, copy and delete the images you have selected.")
    # Use the filter_dataframe function on your dataframe
    filtered_df = pvision.filter_dataframe(st.session_state.df)

    # Display the filtered dataframe
    st.dataframe(filtered_df)

    if st.checkbox("Show filtered images", help=f"This will show the images in your active filter, if you haven't pressed 'add filters' yet, you will not have filtered anything, thus you will see all images in your directory (Current active directory: {st.session_state.directory})"):
        images = []
        captions = []
        for i,row in filtered_df.iterrows():
            images.append(row.filename)
            captions.append(row.positive_prompt)

        img = image_select(
            label = "Select an image",
            images = images,
            captions = captions,
            return_value = "index",
        )

    destination = st.text_input("Directory to move/copy images to")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Copy images in filtered df"):
            pvision.copy_images(filtered_df, destination)
            helper.reset_cached_images()
            return
    with col2:
        if st.button("Move images in filtered df"):
            pvision.move_images(filtered_df, destination)
            helper.reset_cached_images()
            return
    with col3:
        # code to show the item
        submitted = st.button("Delete 🚨", type="secondary")

        if submitted:
            st.session_state.submitted = True

        if "submitted" in st.session_state and st.session_state.submitted:
            st.error("🚨 You are about to delete the images in your filter. Please confirm if this is correct. 🚨")
            confirmation = st.button("Confirm", type="primary")

            if confirmation:
                st.session_state.confirmation = True

        if "confirmation" in st.session_state and st.session_state.confirmation:
            # code to delete the item
            st.success("Item deleted successfully")
            # Delete the images from the filtered dataframe
            pvision.delete_images(filtered_df)
            st.session_state["confirmation"] = False
            st.session_state["submitted"] = False
            # Reset the cache of the images
            helper.reset_cached_images()
            # Return from the function
            return
        
filter_view()