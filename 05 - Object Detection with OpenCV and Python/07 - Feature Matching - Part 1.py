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
# # Feature Matching - Part 1

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
# ## Loading Images

# %%
reeses = cv2.imread('../DATA/reeses_puffs.png', 0)

# %%
display(reeses)

# %%
cereals = cv2.imread('../DATA/many_cereals.jpg', 0)

# %%
display(cereals)

# %% [markdown]
# ## Brute-Force Matching with ORB

# %%
orb = cv2.ORB_create()

# %%
orb

# %%
kp1, des1 = orb.detectAndCompute(reeses, None)

# %%
kp2, des2 = orb.detectAndCompute(cereals, None)

# %%
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# %%
matches = bf.match(des1, des2)

# %%
matches

# %%
matches[0].distance

# %%
matches = sorted(matches, key=lambda x: x.distance)

# %%
len(matches)

# %%
reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)

# %%
display(reeses_matches)
