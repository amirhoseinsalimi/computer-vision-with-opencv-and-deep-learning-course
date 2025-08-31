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
# # Contour Detection

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Loading the Image

# %%
img = cv2.imread('../DATA/internal_external.png', 0)

# %%
img.shape

# %%
plt.imshow(img, cmap='gray')

# %% [markdown]
# ## Finding Contours

# %%
image_copy = img.copy()

contours, hierarchy = cv2.findContours(image_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# %%
plt.imshow(image_copy, cmap='gray')

# %%
type(contours)

# %%
len(contours)

# %%
type(hierarchy)

# %%
len(hierarchy[0])

# %%
img.shape

# %% [markdown]
# ### External Contours

# %%
external_contours = np.zeros(image_copy.shape)

# %%
external_contours.shape

# %%
plt.imshow(image_copy, cmap='gray')

# %%
for i in range(len(contours)):
    # External Contour
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255, -1)

# %%
plt.imshow(external_contours, cmap='gray')

# %% [markdown]
# ### Internal Contours

# %%
internal_contours = np.zeros(image_copy.shape)

for i in range(len(contours)):
    # Internal Contour
    if not hierarchy[0][i][3] == -1:
        cv2.drawContours(internal_contours, contours, i, 255, -1)

# %%
plt.imshow(internal_contours, cmap='gray')
