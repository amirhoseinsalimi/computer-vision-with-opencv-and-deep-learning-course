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
# # Direct Drawing on Images with a mouse - Part 3
#

# %% [markdown]
# ##  Part 3 - Draw Rectangles Using Mouse Dragging

# %%
import cv2
import numpy as np

WINDOW_NAME = 'my_drawing'

drawing = False
ix, iy = -1, -1

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        

cv2.namedWindow(winname=WINDOW_NAME)

cv2.setMouseCallback(WINDOW_NAME, draw_circle)

img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

while True:
    cv2.imshow(WINDOW_NAME, img)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()
