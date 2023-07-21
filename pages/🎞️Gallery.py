import streamlit as st
from PIL import Image
import numpy as np
from streamlit_image_select import image_select
from streamlit_extras.switch_page_button import switch_page

st.session_state.render_image = False

# Get the dataframe from the session state
df = st.session_state["df"]

if df.empty:
    st.warning("Current view is empty, enter new directory")
    st.stop()

with st.container():
    col1, col2,_ = st.columns(3)
    with col1:
        view_selected_image = st.button("View selected image in image viewer")
    with col2:
        st.info("When you click on an image, that image is set as the active image in the image viewer. This means you can either click the button above or just click on '🖼️Image viewer' in the sidebar")

if view_selected_image:
    switch_page("image viewer")

images = []
captions = []
for i,row in df.iterrows():
    images.append(row.filename)
    captions.append(row.positive_prompt)

img = image_select(
    label = "Select an image",
    images = images,
    captions = captions,
    return_value = "index", 
    use_container_width=True,
    index=st.session_state.my_index
)

print(st.session_state.my_index)
st.session_state.my_index = img
print(st.session_state.my_index)
