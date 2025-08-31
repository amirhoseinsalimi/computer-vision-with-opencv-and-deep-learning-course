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
#     version: 3.6.10
# ---

# %% [markdown]
# # Deep Learning on Custom Images - Part 1

# %% [markdown]
# ## Import Libraries

# %%
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Inspect the Data

# %%
cat4 = cv2.imread('../DATA/CATS_DOGS/train/CAT/4.jpg')

cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)

plt.imshow(cat4);

# %%
cat4.shape

# %%
dog2 = cv2.imread('../DATA/CATS_DOGS/train/DOG/2.jpg')

dog2 = cv2.cvtColor(dog2, cv2.COLOR_BGR2RGB)

plt.imshow(dog2);

# %%
dog2.shape

# %% [markdown]
# Note that the shape of the images is different.

# %% [markdown]
# ## Image Data Generator

# %%
dog2.max()

# %%
image_generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# %%
plt.imshow(image_generator.random_transform(dog2))

# %%
image_generator.flow_from_directory('../DATA/CATS_DOGS/train/')
