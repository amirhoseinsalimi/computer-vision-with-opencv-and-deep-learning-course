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
# # Histograms - Part 3 - Histogram Eqaulization

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

# %% [markdown]
# ## Equalize Histograms

# %% [markdown]
# ### Gray Scale Images

# %%
gorilla = cv2.imread('../DATA/gorilla.jpg', 0)


# %%
def display_img(img, cmap=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# %%
display_img(gorilla, 'gray')

# %%
gorilla.shape

# %%
hist_values = cv2.calcHist([gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# %%
plt.plot(hist_values)

# %%
eq_gorilla = cv2.equalizeHist(gorilla)

# %%
display_img(eq_gorilla, cmap='gray')

# %%
hist_values = cv2.calcHist([eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# %%
plt.plot(hist_values)

# %% [markdown]
# ### Color Images

# %%
color_gorilla = cv2.imread('../DATA/gorilla.jpg')

# %%
show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)

# %%
display_img(show_gorilla)

# %%
hsv_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)

# %%
hsv_gorilla[:, :, 2]

# %%
hsv_gorilla[:, :, 2].min()

# %%
hsv_gorilla[:, :, 2].max()

# %%
hsv_gorilla[:, :, 2] = cv2.equalizeHist(hsv_gorilla[:, :, 2])

# %%
eq_color_gorilla = cv2.cvtColor(hsv_gorilla, cv2.COLOR_HSV2RGB)

# %%
display_img(eq_color_gorilla)
