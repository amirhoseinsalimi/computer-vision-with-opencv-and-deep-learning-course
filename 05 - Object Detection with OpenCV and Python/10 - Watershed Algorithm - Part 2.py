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
# # Watershel Algorithm - Part 2 

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
img = cv2.imread('../DATA/pennies.jpg')

# %%
display(img)

# %% [markdown]
# ## Segment Coins Using Watershed Algorithm

# %% [markdown]
# ### 1. Median Blur

# %%
org = img.copy()

img = cv2.medianBlur(img, 35)

# %%
display(img)

# %% [markdown]
# ### 2. Gray Scale

# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# %%
display(gray)

# %% [markdown]
# ### 3. Binary Threshold

# %%
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# %%
display(thresh)

# %% [markdown]
# ### 4. Noise Removal (Optional)

# %%
kernel = np.ones((3, 3), np.uint8)

# %%
kernel

# %%
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# %%
display(opening)

# %% [markdown]
# ### 5. Finding the Background That We Are Sure of

# %%
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# %%
display(sure_bg)

# %% [markdown]
# ### 6. Distance Transform

# %%
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# %%
display(dist_transform)

# %% [markdown]
# ### 7. Thresholding

# %%
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# %%
display(sure_fg)

# %% [markdown]
# ### 8. Find Unknown Region

# %%
sure_fg = np.uint8(sure_fg)

# %%
unkown = cv2.subtract(sure_bg, sure_fg)

# %%
display(unkown)

# %% [markdown]
# ### 9. Label Markers of Sure Foreground

# %%
ret, markers = cv2.connectedComponents(sure_fg)

# %%
markers

# %%
markers = markers + 1

# %%
markers[unkown == 255] = 0

# %%
display(markers)

# %% [markdown]
# ### 10. Apply Watershed Algorithm to find Markers

# %%
markers = cv2.watershed(img, markers)

# %%
display(markers)

# %% [markdown]
# ### 11. Find Contours

# %%
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(org, contours, i, (255, 0, 0), 10)

# %%
display(org)
