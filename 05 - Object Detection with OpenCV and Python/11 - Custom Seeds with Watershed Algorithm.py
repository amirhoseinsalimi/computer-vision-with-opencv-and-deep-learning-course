# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.11
# ---

# %% [markdown]
# # Custom Seeds with Watershed Algorithm

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Loading the Image

# %%
road = cv2.imread('../DATA/road_image.jpg')

# %%
road_copy = np.copy(road)

# %%
plt.imshow(road)

# %%
road.shape

# %% [markdown]
# ## Preparing the Canvas

# %%
road.shape[:2]

# %%
marker_image = np.zeros(road.shape[:2], dtype=np.int32)

# %%
segments = np.zeros(road.shape, dtype=np.uint8)

# %%
marker_image.shape

# %%
segments.shape

# %% [markdown]
# ## Color Mappings

# %%
from matplotlib import cm

# %%
cm.tab10

# %%
cm.tab10(0)

# %%
np.array(cm.tab10(0)[:3]) * 255


# %%
def create_rgb(i):
    x = np.array(cm.tab10(i))[:3] * 255
    
    return tuple(x)


# %%
colors = []

for i in range(10):
    colors.append(create_rgb(i))

# %%
colors

# %% [markdown]
# ## Creating Callbacks and the Windows

# %%
n_markers = 10
current_marker = 1
marks_updated = False


# %%
def mouse_callback(event, x, y, flags, param):
    global marks_updated 

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)
        
        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)
        marks_updated = True


# %%
cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image', mouse_callback)

while True:
    cv2.imshow('WaterShed Segments', segments)
    cv2.imshow('Road Image', road_copy)

    k = cv2.waitKey(1)

    if k == 27:
        break
        
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[0:2], dtype=np.int32)
        segments = np.zeros(road.shape, dtype=np.uint8)
        
    elif k > 0 and chr(k).isdigit():
        current_marker  = int(chr(k))
        
    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)
        
        segments = np.zeros(road.shape, dtype=np.uint8)
        
        for color_ind in range(n_markers):
            segments[marker_image_copy == (color_ind)] = colors[color_ind]
        
        marks_updated = False
        
cv2.destroyAllWindows()
