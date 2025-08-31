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
# # Histograms - Part 1

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
plt.imshow(show_horse)

# %%
plt.imshow(show_rainbow)

# %%
plt.imshow(show_bricks)

# %% [markdown]
# ## Drawing Histograms

# %%
hist_values = cv2.calcHist([blue_bricks], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# %%
hist_values.shape

# %%
plt.plot(hist_values)

# %%
hist_values = cv2.calcHist([dark_horse], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# %%
hist_values.shape

# %%
plt.plot(hist_values)

# %%

# %%
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([blue_bricks], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
    
plt.title('HISTOGRAM FOR BLUE BRICKS')

# %%
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([dark_horse], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 50])
    plt.ylim([0, 500000])
    
plt.title('HISTOGRAM FOR DARK HORSE')
