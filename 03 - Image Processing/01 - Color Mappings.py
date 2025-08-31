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
# # Color Mappings

# %% [markdown]
# ## Converting Colors

# %%
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
img = cv2.imread('../DATA/00-puppy.jpg')

# %%
plt.imshow(img)

# %%
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# %%
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(img)

# %%
img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.imshow(img)
