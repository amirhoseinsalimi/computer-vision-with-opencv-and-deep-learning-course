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
# # Corner Detection - Part 1 - Harris Corner Detection

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
# ## Harris Corner Detection

# %% [markdown]
# ### Flat Image

# %%
gray_flat_chess

# %%
gray = np.float32(gray_flat_chess)

# %%
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

# %%
dst

# %%
plt.imshow(dst)

# %%
dst = cv2.dilate(dst, None)

# %%
plt.imshow(dst)

# %%
flat_chess[dst > 0.01 * dst.max()] = [255, 0, 0]

# %%
plt.imshow(flat_chess)

# %% [markdown]
# ### Real Image

# %%
gray = np.float32(gray_real_chess)

# %%
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

# %%
dst = cv2.dilate(dst, None)

# %%
plt.imshow(dst)

# %%
real_chess[dst > 0.01 * dst.max()] = [255, 0, 0]

# %%
plt.imshow(real_chess)
