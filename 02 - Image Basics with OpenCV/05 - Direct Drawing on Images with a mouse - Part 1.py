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
# # Direct Drawing on Images with a mouse - Part 1

# %% [markdown]
# ##  Part 1 - Connecting an Event Callback to a Window

# %%
import cv2
import numpy as np

WINDOW_NAME = 'my_drawing'

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), radius=100, color=(0, 255, 0), thickness=-1)

cv2.namedWindow(winname=WINDOW_NAME)

cv2.setMouseCallback(WINDOW_NAME, draw_circle)

img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

while True:
    cv2.imshow(WINDOW_NAME, img)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()
