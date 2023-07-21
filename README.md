# PromptvisionNG

An evolution of the previous Promptvision. Rewritten to use Streamlit for its UI. The backend engine has been rewritten. The performance is much better.

The external image scoring model has been changed to ImageReward (https://github.com/THUDM/ImageReward) which is a much better one than the previous model used.

## Features
- ImageReward score scoring of images
- Rating and favoriting images
- Filtering your images on all properties:
	- Prompts
	- Score
	- Personal rating
	- Favorite
	- Resolution
- Copying, moving and deletion of images
- Folder and subfolder navigation (will detect subfolders and enable you to view e.g. all images in a certain folder)

## Setup and installation

1. `git clone https://github.com/Automaticism/Promptvision.git`
2. `cd Promptvision`
3. `conda create -n promptvision`
4. `conda activate promptvision`
5. `pip install -r requirements.txt`
6. `streamlit run Promptvision.py`
7. `Optionally if you use "Calculate ImageReward score" the ImageReward score will be downloaded. This will look like the following in your console:`
	- `2023-07-22 00:56:36.538 Created a temporary directory at /var/folders/l5/n6jwr6tn71s_pm5dh926hbdr0000gn/T/tmpqf7yjji1 
	2023-07-22 00:56:36.538 Writing /var/folders/l5/n6jwr6tn71s_pm5dh926hbdr0000gn/T/tmpqf7yjji1/_remote_module_non_scriptable.py
	Downloading ImageReward.pt:   1%|▌                                             | 21.0M/1.79G [00:02<02:49, 10.4MB/s]
	`

## Application views
- `Promptvision.py` the entrypoint of the multi-page Streammlit app (used to launch the program, `streamlit run Promptvision.py`). This page also contains settings for the application. Set the directory which will be used as the main directory. (E.g. open your main image folder and it will index all images and subfolders. You can then browse every subfolder.)
- `Gallery.py` the gallery view of your images, select one image here and you can look closer in the `Image viewer.py` page
- `Prompt Explorer.py` is running some Natural Language Processing models generating some statistical information to give insight into prompts (This will be the focus for further development)
- `Image viewer.py` is the image viewer. Has typical option of navigating your images (next, previous), shows generational information (supports all the UIs via the https://github.com/d3x-at/sd-parsers module), enables you to set your personal rating and set favorite status, also enables you to view the ImageReward score. Rating & favorite can be used when you are filtering your images (e.g. filter all favorites and copy them to a new folder)

## Bugs and known issues
- This application is limited by Streamlit in some ways with how things are done the Streamlit way (and I am no Streamlit expert), you will get warnings such as 
> AttributeError: st.session_state has no attribute "df". Did you forget to initialize it? More info: https://docs.streamlit.io/library/advanced-features/session-state#initialization

These are typical and will disappear when you set directory and browse the different pages. If they persist and something is broken, create an issue for it.
- `score` might be 0.0 even after installing ImageReward model and having checked off the "Calculate ImageReward score", simply press "Reset cache" and it will recalc and most likely fix the problem

## Credits
- ImageReward https://github.com/THUDM/ImageReward
- Image select for streamlit https://github.com/jrieke/streamlit-image-select
- Streamlit
- Streamlit extras
- sd-parsers https://github.com/d3x-at/sd-parsers