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

# %% [markdown]
# # Image Basics Assessment
#
# ## Complete the Tasks in bold below. Keep in mind, you may need to run some of these tasks as Python scripts.

# %% [markdown]
# ----------
# #### TASK: Open the *dog_backpack.jpg* image from the DATA folder and display it in the notebook. Make sure to correct for the RGB order.

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

# %%
img = cv2.imread('../DATA/dog_backpack.jpg')

# %%
plt.imshow(img);

# %%
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img);

# %% [markdown]
# #### TASK: Flip the image upside down and display it in the notebook.

# %%
img_copy = img.copy()

# %%
img_flipped = cv2.flip(img_copy, 0)

plt.imshow(img_flipped)

# %% [markdown]
# #### TASK: Draw an empty RED rectangle around the dogs face and display the image in the notebook.

# %%
cv2.rectangle(img_copy, pt1=(230, 380), pt2=(590, 740), color=(255, 0, 0), thickness=4)

plt.imshow(img_copy)

# %% [markdown]
# #### TASK: Draw a BLUE TRIANGLE in the middle of the image. The size and angle is up to you, but it should be a triangle (three sides) in any orientation.

# %%
img_copy = img.copy()

vertices = np.array([[460, 380], [230, 740], [590, 740]])

pts = vertices.reshape((-1, 1, 2))

cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=4)

plt.imshow(img_copy)

# %% [markdown]
# ### BONUS TASK. Can you figure our how to fill in this triangle? It requires a different function that we didn't show in the lecture! See if you can use google search to find it.
#
# [CLICK ME FOR A DIRECT LINK TO THE HINT](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html#fillpoly)

# %%
img_copy = img.copy()

vertices = np.array([[460, 380], [230, 740], [590, 740]])

pts = vertices.reshape((-1, 1, 2))

cv2.fillPoly(img_copy, [pts], color=(0, 255, 0))

plt.imshow(img_copy)

# %% [markdown]
# #### TASK: (NOTE: YOU WILL NEED TO RUN THIS AS A SCRIPT). Create a script that opens the picture and allows you to draw empty red circles whever you click the RIGHT MOUSE BUTTON DOWN.

# %%
WINDOW_NAME = 'Window'

img_copy = img.copy()

def handler(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img_copy, (x, y), 40, (0, 0, 255), 4)

cv2.namedWindow(winname=WINDOW_NAME)

cv2.setMouseCallback(WINDOW_NAME, handler)

while True:
    cv2.imshow(WINDOW_NAME, img_copy)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()
