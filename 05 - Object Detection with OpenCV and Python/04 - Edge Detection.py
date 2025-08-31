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
# # Edge Detection

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
img = cv2.imread('../DATA/sammy_face.jpg')

# %%
plt.imshow(img)

# %%
edges = cv2.Canny(image=img, threshold1=127, threshold2=127)

# %%
plt.imshow(edges)

# %%
edges = cv2.Canny(image=img, threshold1=0, threshold2=255)

# %%
plt.imshow(edges)

# %%
median_value = np.median(img)

# %%
median_value

# %%
# LOWER THRESHOLD TO EITHER 0 OR 70% OF THE MEDAIN VALUE WHICHEVER IS GREATER
lower = int(max(0, 0.7 * median_value))

# UPPER THRESHOLD  TO EITHER 130% OF THE MEDIAN OR THE MAX VALUE OF 255 WHICHEVER IS SMALLER
upper = int(min(255, 1.3 * median_value))

# %%
lower, upper

# %%
edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper)

# %%
plt.imshow(edges)

# %%
edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper + 100)

# %%
plt.imshow(edges)

# %%
blurred_img = cv2.blur(img, ksize=(5, 5))

# %%
edges = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper + 50)

# %%
plt.imshow(edges)
