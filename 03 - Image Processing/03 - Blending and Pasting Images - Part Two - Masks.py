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
# # Blending and Pasting Images - Part 2 - Masks

# %% [markdown]
# ## Loading Images

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline

# %%
img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# %%
plt.imshow(img1)

# %%
plt.imshow(img2)

# %% [markdown]
# ## Resizing Images

# %%
img2 = cv2.resize(img2, (600, 600))

# %%
plt.imshow(img2)

# %%
img1.shape

# %%
img2.shape

# %%
x_offset = 934 - 600
y_offset = 1401 - 600

# %%
rows, cols, channels = img2.shape

# %%
rows

# %%
cols

# %%
channels

# %%
roi = img1[y_offset:1401, x_offset:943]

# %%
plt.imshow(roi)

# %% [markdown]
# ## Create a Mask

# %%
img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# %%
plt.imshow(img2gray, cmap='gray')

# %%
mask_inv = cv2.bitwise_not(img2gray)

# %%
plt.imshow(mask_inv, cmap='gray')

# %%
mask_inv.shape

# %%
white_backgroud = np.full(img2.shape,  255, dtype=np.uint8)

# %%
white_backgroud.shapez

# %%
white_backgroud

# %%
bk = cv2.bitwise_or(white_backgroud, white_backgroud, mask=mask_inv)

# %%
bk.shape

# %%
plt.imshow(bk)

# %%
fg = cv2.bitwise_or(img2, img2, mask=mask_inv)

# %%
plt.imshow(fg)

# %%
final_roi = cv2.bitwise_or(roi, fg)

# %%
plt.imshow(final_roi)

# %%
large_img = img1
small_img = final_roi

# %%
large_img[y_offset:y_offset + small_img.shape[0], x_offset:x_offset + small_img.shape[1]] = small_img

# %%
plt.imshow(large_img)
