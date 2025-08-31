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
#     version: 3.6.11
# ---

# %% [markdown]
# # Optical Flow Coding with OpenCV - Part 1

# %%
import numpy as np
import cv2

# %%
corner_track_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)

# %%
lk_params = dict(winSize=(200, 200), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))

# %%
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read(0)

prvsImg = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(frame1)
hsv_mask[:, :, 1] = 255

while True:
    ret, frame2 = cap.read(0)
    
    nextImage = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvsImg, nextImage, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    hsv_mask[:, :, 0] = ang / 2
    hsv_mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    cv2.imshow('Frame', bgr)
    
    k = cv2.waitKey(10) & 0xFF
    
    if k == 27:
        break
        
    prevImg = nextImage.copy()

cap.release()
cv2.destroyAllWindows()
