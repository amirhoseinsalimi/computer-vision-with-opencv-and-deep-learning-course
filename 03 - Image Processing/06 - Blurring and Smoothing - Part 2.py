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
# # Blurring and Smoothing - Part 2

# %% [markdown]
# ## Loading and Display Images

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
def load_img():
    img = cv2.imread('../DATA/bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


# %%
def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)


# %%
def load_img_with_text():
    img = load_img()
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, text='brick', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)
    
    return img


# %%
i = load_img()
display_img(i)

# %% [markdown]
# ## Gamma Correction

# %%
gamma = 1/4

# %%
result = np.power(i, gamma)
display_img(result)

# %%
gamma = 1/10

# %%
result = np.power(i, gamma)
display_img(result)

# %%
gamma = 2

# %%
result = np.power(i, gamma)
display_img(result)

# %%
gamma = 8

# %%
result = np.power(i, gamma)
display_img(result)

# %% [markdown]
# ## Low-pass Filter

# %%
img = load_img_with_text()

# %%
display_img(img)

# %%
kernel = np.ones(shape=(5, 5)).astype(np.float32) / 25

# %%
kernel

# %%
1 / 25

# %%
dst = cv2.filter2D(img, -1, kernel)
display_img(dst)

# %%
img = load_img_with_text()

# %%
blurred = cv2.blur(img, ksize=(5, 5))

# %%
display_img(blurred)

# %%
blurred = cv2.blur(img, ksize=(10, 10))

# %%
display_img(blurred)

# %% [markdown]
# ## Gaussian Blur

# %%
img = load_img_with_text()

# %%
blurred_img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=10)

# %%
display_img(blurred_img)

# %% [markdown]
# ## Median Blur

# %%
img = load_img_with_text()

# %%
media_result = cv2.medianBlur(img, 5)

# %%
display_img(media_result)

# %% [markdown]
# ## Removing Noise Using Blurring

# %%
img = cv2.imread('../DATA/sammy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# %%
display_img(img)

# %%
noise_img = cv2.imread('../DATA/sammy_noise.jpg')
noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)

# %%
display_img(noise_img)

# %%
median = cv2.medianBlur(noise_img, 5)

# %%
display_img(median)

# %% [markdown]
# ## Bilateral Filtering

# %%
img = load_img_with_text()

# %%
blur = cv2.bilateralFilter(img, 9, 75, 75)

# %%
display_img(blur)
