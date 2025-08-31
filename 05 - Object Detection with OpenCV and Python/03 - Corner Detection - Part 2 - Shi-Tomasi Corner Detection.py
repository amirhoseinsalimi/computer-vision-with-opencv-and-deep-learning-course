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
# # Corner Detection - Part 2 - Shi-Tomasi Corner Detection

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Loading and Displaying Images

# %%
flat_chess = cv2.imread('../DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)

# %%
plt.imshow(flat_chess)

# %%
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

# %%
plt.imshow(gray_flat_chess, cmap='gray')

# %%
real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)

# %%
plt.imshow(real_chess)

# %%
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

# %%
plt.imshow(gray_real_chess, cmap='gray')

# %% [markdown]
# ## Shi-Tomasi Corner Detection

# %% [markdown]
# ### Flat Image

# %% [markdown]
# #### `maxCorners = 5`

# %%
corners = cv2.goodFeaturesToTrack(gray_flat_chess, 5, 0.01, 10)

# %%
corners

# %%
corners = np.int0(corners)

# %%
for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x, y), 3, (255, 0, 0), -1)

# %%
plt.imshow(flat_chess)

# %% [markdown]
# #### `maxCorners = 64`

# %%
corners = cv2.goodFeaturesToTrack(gray_flat_chess, 64, 0.01, 10)

# %%
corners

# %%
corners = np.int0(corners)

# %%
for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x, y), 3, (255, 0, 0), -1)

# %%
plt.imshow(flat_chess)

# %% [markdown]
# ## Real Image

# %% [markdown]
# #### `maxCorners = 80`

# %%
corners = cv2.goodFeaturesToTrack(gray_real_chess, 80, 0.01, 10)

# %%
corners

# %%
corners = np.int0(corners)

# %%
for i in corners:
    x, y = i.ravel()
    cv2.circle(real_chess, (x, y), 3, (255, 0, 0), -1)

# %%
plt.imshow(real_chess)

# %% [markdown]
# #### `maxCorners = 100`

# %%
corners = cv2.goodFeaturesToTrack(gray_real_chess, 100, 0.01, 10)

# %%
corners

# %%
corners = np.int0(corners)

# %%
for i in corners:
    x, y = i.ravel()
    cv2.circle(real_chess, (x, y), 3, (255, 0, 0), -1)

# %%
plt.imshow(real_chess)
