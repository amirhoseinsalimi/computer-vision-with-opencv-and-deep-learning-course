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
# # Drawing on Images - Part 1 - Basic Shapes

# %% [markdown]
# ## Drawing Shapes

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
# ### Rectangles

# %%
cv2.rectangle(blank_img, pt1=(384, 10), pt2=(500, 150), color=(0, 255, 0), thickness=10)

# %%
plt.imshow(blank_img)

# %% [markdown]
# ### Squares

# %%
cv2.rectangle(blank_img, pt1=(200, 200), pt2=(300, 300), color=(0, 0, 255), thickness=10)

# %%
plt.imshow(blank_img)

# %% [markdown]
# ### Circles

# %%
cv2.circle(blank_img, center=(100, 100), radius=50, color=(255, 0, 0), thickness=8)

# %%
plt.imshow(blank_img)

# %% [markdown]
# ### Filled Circles

# %%
cv2.circle(blank_img, center=(400, 400), radius=50, color=(255, 0, 0), thickness=-1)

# %%
plt.imshow(blank_img)

# %% [markdown]
# ### Lines

# %%
cv2.line(blank_img, pt1=(0, 0), pt2=(512, 512), color=(102, 255, 255), thickness=5)

# %%
plt.imshow(blank_img)
