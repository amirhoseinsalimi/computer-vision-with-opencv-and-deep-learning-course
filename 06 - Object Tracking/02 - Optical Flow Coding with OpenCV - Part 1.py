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

ret, prev_frame = cap.read(0)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)

mask = np.zeros_like(prev_frame, dtype=np.uint8)

while True:
    ret, frame = cap.read(0)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)
    good_new = nextPts[status==1]
    good_prev = prevPts[status==1]
    
    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        
        mask = cv2.line(mask, (int(x_new), int(y_new)), (int(x_prev), int(y_prev)), (0, 255, 0), 3)
        
        frame = cv2.circle(frame, (int(x_new), int(y_new)), 8, (0, 0, 255), -1)
    
    img = cv2.add(frame, mask)
    cv2.imshow('Tracing', img)
    
    k = cv2.waitKey(30) & 0xFF
    
    if k == 27:
        break
        
    prev_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 1)

cap.release()
cv2.destroyAllWindows()
