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
# # NumPy and Image Assessment 
#
# **COMPLETE THE TASKS IN BOLD BELOW.**

# %% [markdown]
# **TASK: Import NumPy**

# %%
import numpy as np

# %% [markdown]
# **TASK: Create a 5 by 5 array where every number is a 10**

# %%
arr = np.zeros(shape=(5, 5))

# %%
arr[:, :] = 10
arr

# %% [markdown]
# **TASK: Run the cell below to create an array of random numbers and see if you can figure out how it works.**

# %%
# This line sets a "seed" so you get the same random numbers we do
np.random.seed(101)
# This line creates an array of random numbers
arr = np.random.randint(low=0, high=100, size=(5, 5))

# %%
arr

# %% [markdown]
# **TASK: What are the largest and smallest values in this array?**

# %%
largest = arr.max()
largest

# %%
smallest = arr.min()
smallest

# %% [markdown]
# **TASK: Use PIL and matplotlib to read and display the ../DATA/00-puppy.jpg image.**

# %%
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib inline

pic = Image.open('../DATA/00-puppy.jpg')

pic

# %% [markdown]
# **TASK: Convert the image to a NumPy Array**

# %%
pic_arr = np.asarray(pic)

pic_arr.shape

# %% [markdown]
# **FINAL TASK: Use slicing to set the RED and GREEN channels of the picture to 0, then use imshow() to show the isolated blue channel**

# %%
pic_blue = pic_arr.copy()

# %%
pic_blue[:, :, 0] = 0
pic_blue[:, :, 1] = 0

# %%
plt.imshow(pic_blue)

# %% [markdown]
# ## Great Job!

# %% [markdown]
# ----
