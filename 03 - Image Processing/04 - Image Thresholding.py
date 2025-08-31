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
# # Image Thresholding

# %% [markdown]
# ## Loading Images

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline

# %%
img = cv2.imread('../DATA/rainbow.jpg')

# %%
plt.imshow(img)

# %% [markdown]
# ## Thresholding

# %%
img = cv2.imread('../DATA/rainbow.jpg', 0)

# %%
img.shape

# %%
plt.imshow(img, cmap='gray')

# %%
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# %%
ret

# %%
thresh1

# %%
plt.imshow(thresh1, cmap='gray')

# %%
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# %%
plt.imshow(thresh1, cmap='gray')

# %%
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

# %%
plt.imshow(thresh1, cmap='gray')

# %%
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

# %%
plt.imshow(thresh1, cmap='gray')

# %%
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

# %%
plt.imshow(thresh1, cmap='gray')

# %% [markdown]
# ## Dynamic Thresholding

# %%
img = cv2.imread('../DATA/crossword.jpg', 0)

# %%
plt.imshow(img, cmap='gray')


# %%
def show_pic(img):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# %%
show_pic(img)

# %%
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
show_pic(th1)

# %%
ret, th1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
show_pic(th1)

# %%
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(th2)

# %%
blended = cv2.addWeighted(th1, 0.6, th2, 0.4, 0)

# %%
show_pic(blended)
