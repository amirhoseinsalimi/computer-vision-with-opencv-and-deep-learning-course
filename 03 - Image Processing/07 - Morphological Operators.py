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
# # Morphological Operators

# %% [markdown]
# ## Loading and Displaying Images

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
def load_img():
    blank_img = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='ABCDE', org=(50, 300), fontFace=font, fontScale=5, color=(255, 255, 255), thickness=25)
    
    return blank_img


# %%
def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# %%
img = load_img()
display_img(img)

# %% [markdown]
# ## Erosion

# %%
kernel = np.ones(shape=(5, 5), dtype=np.int8)

# %%
kernel

# %%
result = cv2.erode(img, kernel, iterations=1)

# %%
display_img(result)

# %%
img = load_img()
display_img(img)

# %%
result = cv2.erode(img, kernel, iterations=4)

# %%
display_img(result)

# %% [markdown]
# ## Openning

# %%
img = load_img()

# %%
white_noise = np.random.randint(low=0, high=2, size=(600, 600))

# %%
white_noise

# %%
display_img(white_noise)

# %%
img.max()

# %%
white_noise = white_noise * 255

# %%
white_noise

# %%
display_img(white_noise)

# %%
noise_img = white_noise + img

# %%
display_img(noise_img)

# %%
opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)

# %%
display_img(opening)

# %% [markdown]
# ## Closing

# %%
img = load_img()

# %%
black_noise = np.random.randint(low=0, high=2, size=(600, 600))

# %%
black_noise = black_noise * -255

# %%
black_noise

# %%
display_img(black_noise)

# %%
black_noise_img = img + black_noise

# %%
black_noise_img

# %%
black_noise_img[black_noise == -255] = 0

# %%
black_noise_img.min()

# %%
display_img(black_noise_img)

# %%
closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)

# %%
display_img(closing)

# %% [markdown]
# ## Gradient

# %%
img = load_img()

# %%
display_img(img)

# %%
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# %%
display_img(gradient)
