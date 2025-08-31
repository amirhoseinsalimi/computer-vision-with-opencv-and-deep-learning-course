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
# <a href="https://www.pieriandata.com"><img src="../DATA/Logo.jpg"></a>
# *Copyright Pierian Data Inc.*

# %% [markdown]
# # Image Processing Assessment
#
# Complete the tasks in bold below! Feel free to treat this as a code along by skipping this assessment lecture and moving straight to the solutions lecture!

# %% [markdown]
# **Some Useful Code is already here for you in the cells below:**

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

# %%
def display_img(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)


# %% [markdown]
# **TASK: Open and display the giraffes.jpg image that is located in the DATA folder.**

# %%
giraffes = cv2.imread('../DATA/giraffes.jpg')
giraffes = cv2.cvtColor(giraffes, cv2.COLOR_BGR2RGB)

# %%
display_img(giraffes)

# %% [markdown]
# **TASK:Apply a binary threshold onto the image.**

# %%
giraffes_gray = cv2.cvtColor(giraffes, cv2.COLOR_RGB2GRAY)
ret, thr1 = cv2.threshold(giraffes_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

# %%
thr1.shape

# %%
display_img(thr1, cmap='gray')

# %% [markdown]
# **TASK: Open the giraffes.jpg file from the DATA folder and convert its colorspace to  HSV and display the image.**

# %%
giraffes_hsv = cv2.imread('../DATA/giraffes.jpg')
giraffes_hsv = cv2.cvtColor(giraffes_hsv, cv2.COLOR_BGR2HSV)

# %%
display_img(giraffes_hsv)

# %% [markdown]
# **TASK: Create a low pass filter with a 4 by 4 Kernel filled with values of 1/10 (0.01) and then use 2-D Convolution to blur the giraffes image (displayed in normal RGB)**

# %%
kernel = np.ones((4, 4), dtype=np.float32) / 10

# %%
kernel

# %%
dest = cv2.filter2D(giraffes, -1, kernel)

# %%
display_img(dest)

# %% [markdown]
# **TASK: Create a Horizontal Sobel Filter (sobelx from our lecture) with a kernel size of 5 to the grayscale version of the giraffes image and then display the resulting gradient filtered version of the image.**

# %%
sobelx = cv2.Sobel(giraffes_gray, cv2.CV_64F, 1, 0, ksize=5)

# %%
sobelx

# %%
display_img(sobelx, cmap='gray')

# %% [markdown]
# **TASK: Plot the color histograms for the RED, BLUE, and GREEN channel of the giraffe image. Pay careful attention to the ordering of the channels.**

# %%
giraffes = cv2.imread('../DATA/giraffes.jpg')

channels = ('b', 'g', 'r')

for i, channel in enumerate(channels):
    histr = cv2.calcHist([giraffes], [i], None, [256], [0, 256])
    plt.plot(histr, color=channel)
    plt.xlim([0, 256])

plt.title('HISTOGRAM FOR GIRAFFES')

# %% [markdown]
# # Great job!
