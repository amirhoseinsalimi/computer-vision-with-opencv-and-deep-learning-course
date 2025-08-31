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
# # Histograms - Part 2 - Histogram Eqaulization

# %% [markdown]
# ## Loading and Showing Images

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
dark_horse = cv2.imread('../DATA/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('../DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)

# %%
img = rainbow

# %%
img.shape

# %%
img.shape[:2]

# %% [markdown]
# ## Create a Mask

# %%
mask = np.zeros(img.shape[:2], np.uint8)

# %%
plt.imshow(mask, cmap='gray')

# %%
mask[300:400, 100:400] = 255

# %%
plt.imshow(mask, cmap='gray')

# %%
plt.imshow(show_rainbow)

# %%
masked_img = cv2.bitwise_and(img, img, mask=mask)

# %%
plt.imshow(masked_img)

# %%
show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, mask=mask)

# %%
plt.imshow(show_masked_img)

# %% [markdown]
# ## Plotting Histograms

# %%
hist_masked_values_red = cv2.calcHist([rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0, 256])

# %%
hist_values_red = cv2.calcHist([rainbow], channels=[2], mask=None, histSize=[256], ranges=[0, 256])

# %%
plt.plot(hist_masked_values_red)
plt.title('Red Histogram for masked rainbow')

# %%
plt.plot(hist_values_red)
plt.title('Red Histogram for normal rainbow')
