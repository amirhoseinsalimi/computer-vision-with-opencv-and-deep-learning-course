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
# # Grandients

# %% [markdown]
# ## Loading and Displaying Images

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# %%
img = cv2.imread('../DATA/sudoku.jpg', 0)

# %%
display_img(img)

# %% [markdown]
# ## Sobel-Feldman Operator

# %% [markdown]
# ### X Axis

# %%
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

# %%
display_img(sobelx)

# %% [markdown]
# ### Y Axis

# %%
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# %%
display_img(sobely)

# %% [markdown]
# ### X and Y Axes

# %%
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)

# %%
display_img(sobelxy)

# %% [markdown]
# ## Laplacian Derivatives

# %%
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# %%
display_img(laplacian)

# %%
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)

# %%
display_img(laplacian)

# %% [markdown]
# ## Blending Soble on X Axis and Y Axis

# %%
blended = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, gamma=0)

# %%
display_img(blended)

# %% [markdown]
# ## Thresholding

# %%
ret, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# %%
display_img(th1)

# %% [markdown]
# ## Gradient Morphological Operator

# %%
kernel = np.ones((4, 4), np.uint8)

# %%
gradient = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)

# %%
display_img(gradient)
