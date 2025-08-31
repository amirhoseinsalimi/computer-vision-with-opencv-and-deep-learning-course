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
# # Watershed Algorithm - Part 1

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
def display(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# %% [markdown]
# ## Loading and Displaying the Image

# %%
sep_coins = cv2.imread('../DATA/pennies.jpg')

# %%
display(sep_coins)

# %% [markdown]
# ## Segment Coins Using Previous Knowledge

# %% [markdown]
# ### Median Blur

# %%
sep_blur = cv2.medianBlur(sep_coins, 25)

# %%
display(sep_blur)

# %% [markdown]
# ### Gray Scale

# %%
gray_sep_coins = cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)

# %%
display(gray_sep_coins)

# %% [markdown]
# ### Binary Threshold

# %%
ret, sep_thresh = cv2.threshold(gray_sep_coins, 160, 255, cv2.THRESH_BINARY_INV)

# %%
display(sep_thresh)

# %% [markdown]
# ### Finding Contours

# %%
image = sep_thresh.copy()

contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# %%
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255, 0, 0), 10)

# %%
display(sep_coins)
