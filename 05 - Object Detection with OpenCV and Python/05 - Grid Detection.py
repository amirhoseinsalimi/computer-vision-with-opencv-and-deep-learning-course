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
# # Grid Detection

# %%
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Find Chessboard Corners

# %%
flat_chess = cv2.imread('../DATA/flat_chessboard.png')

# %%
plt.imshow(flat_chess)

# %%
found, corners = cv2.findChessboardCorners(flat_chess, (7, 7))

# %%
found, corners

# %%
cv2.drawChessboardCorners(flat_chess, (7, 7), corners, found)

# %%
plt.imshow(flat_chess)

# %% [markdown]
# ## Find Circles Grid

# %%
dots = cv2.imread('../DATA/dot_grid.png')

# %%
plt.imshow(dots)

# %%
found, corners = cv2.findCirclesGrid(dots, (10, 10), cv2.CALIB_CB_SYMMETRIC_GRID)

# %%
found, corners

# %%
cv2.drawChessboardCorners(dots, (10, 10), corners, found)

# %%
plt.imshow(dots)
