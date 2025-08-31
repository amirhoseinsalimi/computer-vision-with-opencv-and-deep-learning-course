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
# # Blending and Pasting Images - Part 1

# %% [markdown]
# ## Loading Images

# %%
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
img1 = cv2.imread('../DATA/dog_backpack.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# %%
plt.imshow(img1)

# %%
plt.imshow(img2)

# %%
img1.shape

# %%
img2.shape

# %% [markdown]
# ## Blending Images of the Same Size

# %%
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200))

# %%
blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)

# %%
plt.imshow(blended)

# %%
blended = cv2.addWeighted(src1=img1, alpha=0.8, src2=img2, beta=0.1, gamma=0)

# %%
plt.imshow(blended)

# %%
blended = cv2.addWeighted(src1=img1, alpha=0.8, src2=img2, beta=0.1, gamma=0.5)

# %%
plt.imshow(blended)

# %% [markdown]
# ## Overlay Small Images on Top of a Larger Image (No Blending)

# %% [markdown]
# ### 1. Numpy Reassignment

# %%
img1 = cv2.imread('../DATA/dog_backpack.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# %%
img2 = cv2.resize(img2, (600, 600))

# %%
plt.imshow(img2)

# %%
plt.imshow(img1)

# %%
large_img = img1
small_img = img2

# %%
x_offset = 0
y_offset = 0

# %%
x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

# %%
small_img.shape

# %%
large_img[y_offset:y_end, x_offset:x_end] = small_img

# %%
plt.imshow(large_img)
