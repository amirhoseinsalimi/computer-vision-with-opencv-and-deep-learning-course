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
# # Feature Matching - Part 2

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
# # Loading Images

# %%
reeses = cv2.imread('../DATA/reeses_puffs.png', 0)

# %%
display(reeses)

# %%
cereals = cv2.imread('../DATA/many_cereals.jpg', 0)

# %%
display(cereals)

# %% [markdown]
# ## Brute-Force Matching with SIFT Descriptors and Ratio Test

# %%
sift = cv2.xfeatures2d.SIFT_create()

# %%
kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

# %%
des1

# %%
bf = cv2.BFMatcher()

# %%
matches = bf.knnMatch(des1, des2, k=2)

# %%
# matches

# %% [markdown]
# ### Ratio Test (less distance == better match)

# %%
good = []

for match1, match2 in matches:
    # If match1 distance is less than 75% of match2 distance
    # then descriptor was a good match, let's keeps it
    if match1.distance < 0.75 * match2.distance:
        good.append([match1])

# %%
len(good), len(matches)

# %%
sift_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags=2)

# %%
display(sift_matches)

# %% [markdown]
# ## FLANN-based Macther

# %%
sift = cv2.xfeatures2d.SIFT_create()

# %%
kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

# %%
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# %%
flann = cv2.FlannBasedMatcher(index_params, search_params)

# %%
matchees = flann.knnMatch(des1, des2, k=2)

# %% [markdown]
# ### Ratio Test

# %%
good = []

for match1, match2 in matches:
    # If match1 distance is less than 70% of match2 distance
    # then descriptor was a good match, let's keeps it
    if match1.distance < 0.70 * match2.distance:
        good.append([match1])

# %%
flann_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags=0)

# %%
display(flann_matches)

# %% [markdown]
# ### FLANN-based Macther wit Mask

# %%
sift = cv2.xfeatures2d.SIFT_create()

# %%
kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

# %%
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# %%
flann = cv2.FlannBasedMatcher(index_params, search_params)

# %%
matchees = flann.knnMatch(des1, des2, k=2)

# %%
matchesMask = [[0, 0] for i in range(len(matches))]

# %%
# matchesMask

# %%
for i, (match1, match2) in enumerate(matches):
    # If match1 distance is less than 70% of match2 distance
    # then descriptor was a good match, let's keeps it
    if match1.distance < 0.70 * match2.distance:
        matchesMask[i] = [1, 0]

# %%
draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=0
)

# %%
flann_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, matches, None, **draw_params)

# %%
display(flann_matches)
