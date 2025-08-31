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
#     version: 3.6.15
# ---

# %% [markdown]
# # Drawing on Images - Part 2 - Text and Polygons

# %% [markdown]
# ## Putting Text and Drawing Polygons

# %%
import cv2
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ### Creating a Blank Image

# %%
blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)

# %%
blank_img.shape

# %%
plt.imshow(blank_img)

# %% [markdown]
# ### Text

# %%
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(blank_img, text='Hello', org=(10, 500), fontFace=font, fontScale=4, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

# %%
plt.imshow(blank_img)

# %% [markdown]
# ### Polygons

# %%
blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int32)

# %%
plt.imshow(blank_img)

# %%
vertices = np.array([
    [100, 300],
    [200, 200],
    [400, 300],
    [200, 400]
], dtype=np.int32)

# %%
vertices

# %%
vertices.shape

# %%
pts = vertices.reshape((-1, 1, 2))

# %%
vertices.shape

# %%
pts.shape

# %%
cv2.polylines(blank_img, [pts], isClosed=True, color=(255, 0, 0), thickness=5)

# %%
plt.imshow(blank_img)
