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
# # Opening Image files in a notebook

# %% [markdown]
# ## Openning Images and Possible Errors

# %%
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
import cv2

# %%
img = cv2.imread('../DATA/00-puppy.jpg')

# %%
type(img)

# %%
img = cv2.imread('wrong_path')

# %%
type(img)

# %%
img = cv2.imread('../DATA/00-puppy.jpg')

# %%
type(img)

# %%
img.shape

# %%
plt.imshow(img)

# %%
# MATPLOTLB -- RGB
# OPENCV -- BGR

# %%
fixed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# %%
plt.imshow(fixed_img)

# %%
img_gray = cv2.imread('../DATA/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)

# %%
img_gray.shape

# %%
img_gray.min()

# %%
img_gray.max()

# %%
plt.imshow(img_gray)

# %%
plt.imshow(img_gray, cmap='gray')

# %%
plt.imshow(img_gray, cmap='magma')

# %% [markdown]
# ## Resizing Images

# %%
plt.imshow(fixed_img)

# %%
fixed_img.shape

# %%
new_img = cv2.resize(fixed_img, (1000, 400))

# %%
plt.imshow(new_img)

# %%
w_ratio = 0.8
h_ratio = 0.2

# %%
new_img = cv2.resize(fixed_img, (0, 0), fixed_img, w_ratio, h_ratio)

# %%
plt.imshow(new_img)

# %%
new_img.shape

# %% [markdown]
# ## Flipping Images

# %%
new_img = cv2.flip(fixed_img, 0)
plt.imshow(new_img)

# %%
new_img = cv2.flip(fixed_img, 1)
plt.imshow(new_img)

# %%
new_img = cv2.flip(fixed_img, -1)
plt.imshow(new_img)

# %% [markdown]
# ## Saving Images

# %%
cv2.imwrite('totally_new.jpg', fixed_img)

# %% [markdown]
# ## Working with Matplotlib Figures

# %%
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)
ax.imshow(fixed_img)
