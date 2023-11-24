import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pvision
from pathlib import Path
from streamlit.errors import StreamlitAPIException
import base64
import helper
from streamlit_extras.switch_page_button import switch_page

st.session_state.widemode = None
helper.store_metadata_in_session(st.session_state.my_index)


def set_new_df_value(key, value):
    # print(f"setting new df values - key:{key} - value:{value}")
    st.session_state["df"].loc[st.session_state.my_index, key] = value
    pvision.save_df_from_streamlit(st.session_state.directory, st.session_state.df)
    helper.store_metadata_in_session(st.session_state.my_index)
    st.session_state["db_updated"] = True


def favorite_render_false():
    set_new_df_value("favorite", True if not st.session_state.favorite else False)


def render_image(container):
    helper.store_metadata_in_session(st.session_state.my_index)
    # Get the selected image and its metadata from the session_state
    filename = st.session_state.filename
    positive_prompt = st.session_state.positive_prompt
    negative_prompt = st.session_state.negative_prompt
    metadata = st.session_state.metadata
    score = st.session_state.score

    with container:
        img_view_col1, img_view_col2 = st.columns(2, gap="large")

        with img_view_col1:
            if st.session_state.widemode:
                header_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='max-height: 80vh;'>".format(
                    helper.img_to_bytes(filename)
                )
                st.markdown(
                    header_html,
                    unsafe_allow_html=True,
                )
            else:
                image = Image.open(filename)
                st.image(
                    image,
                    caption=filename,
                )  # Use column width to fit the image to the browser window
        with img_view_col2:
            st.caption("Positive prompt")
            st.success(positive_prompt)
            # Negative prompt container
            st.caption("Negative prompt")
            st.warning(negative_prompt)
            # Score container
            st.metric(label="ImageReward Score", value=score)
            st.info(
                f"Currently viewing image {st.session_state.my_index} out of {len(st.session_state.df)} in directory: {st.session_state.directory}"
            )

            with st.container():
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.caption("Favorite")
                    st.write(st.session_state.favorite)
                with subcol2:
                    if st.session_state.favorite:
                        favorite = st.button(
                            "Unfavorite", on_click=favorite_render_false
                        )
                    else:
                        favorite = st.button("Favorite", on_click=favorite_render_false)
            subcol1, subcol2 = st.columns(2)
            with subcol2:
                with st.form("my_form", clear_on_submit=True):
                    rating = st.number_input("Rate the image", min_value=0, max_value=5)
                    if st.form_submit_button("Submit"):
                        if rating:
                            set_new_df_value("rating", int(rating))
                        else:
                            set_new_df_value("rating", 0)
            with subcol1:
                st.metric(label="Rating", value=st.session_state.rating)


            # Metadata container
            st.caption("Metadata")
            st.json(metadata)


with st.container():
    # Display the image and its metadata in the main area
    st.title("Promptvision")
    # Create next and previous buttons to change the st.session_state.my_index
    col1, col2, col3, col4, col5 = st.columns(5)
    with col2:
        if st.button("Next"):
            st.session_state.my_index = (st.session_state.my_index + 1) % len(
                st.session_state["df"]
            )  # Increase the st.session_state.my_index by 1 and wrap around to 0 if it reaches the last index
            st.session_state.render_image = False
    with col1:
        if st.button("Previous"):
            st.session_state.my_index = (st.session_state.my_index - 1) % len(
                st.session_state["df"]
            )  # Decrease the st.session_state.my_index by 1 and wrap around to the last index if it reaches 0
            st.session_state.render_image = False
    with col3:
        image_jump_index = st.number_input("View image number: ", min_value=1, max_value=len(st.session_state.df) - 1,
                                           value=st.session_state.my_index)
        if image_jump_index:
            st.session_state.my_index = image_jump_index
    with col4:
        if st.button("Gallery"):
            switch_page("gallery")
    with col5:
        if st.checkbox(
            "Non-wide mode",
            help="Toggle this to scale the image for smaller screens. Will make viewing non-responsive on large screens.",
        ):
            st.session_state.widemode = False
            st.session_state.render_image = False
        else:
            st.session_state.widemode = True
            st.session_state.render_image = False

    image_container = st.container()
    if not st.session_state.render_image:
        render_image(image_container)
        st.session_state.render_image = True
