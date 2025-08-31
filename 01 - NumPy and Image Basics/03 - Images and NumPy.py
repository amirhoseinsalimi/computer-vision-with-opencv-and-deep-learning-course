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
# # Images and NumPy

# %% [markdown]
# ## Working with PIL

# %%
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
from PIL import Image

# %%
pic = Image.open('../DATA/00-puppy.jpg')

# %%
pic

# %%
type(pic)

# %%
pic_arr = np.asarray(pic)

# %%
type(pic_arr)

# %%
pic_arr.shape

# %%
plt.imshow(pic_arr)

# %% [markdown]
# ## Color Channels

# %%
pic_red = pic_arr.copy()

# %%
plt.imshow(pic_red)

# %%
pic_red.shape

# %%
pic_red[:, :, 1]

# %%
plt.imshow(pic_red[:, :, 1])

# %%
plt.imshow(pic_red[:, :, 0])

# %%
# READ CHANNGEL VALUES
plt.imshow(pic_red[:, :, 0], cmap='gray')

# %%
# READ CHANNGEL VALUES
plt.imshow(pic_red[:, :, 1], cmap='gray')

# %%
# READ CHANNGEL VALUES
plt.imshow(pic_red[:, :, 2], cmap='gray')

# %%
# GREEN CHANNEL
pic_red[:, :, 1] = 0

# %%
plt.imshow(pic_red)

# %%
# BLUE CHANNEL
pic_red[:, :, 2] = 0

# %%
plt.imshow(pic_red)

# %%
pic_red.shape
