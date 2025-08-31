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
# # MeanShift and CamShift Tracking with OpenCV 

# %%
import numpy as np
import cv2

# %% [markdown]
# ## MeanShift Tracking

# %%
cap = cv2.VideoCapture(0)

ret, frame = cap.read(0)

face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')

face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0])

track_window = (face_x, face_y, w, h)

roi = frame[face_y:face_y + h, face_x:face_x + w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_histogram = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

cv2.normalize(roi_histogram, roi_histogram, 0, 255, cv2.NORM_MINMAX)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read(0)
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        dest = cv2.calcBackProject([hsv], [0], roi_histogram, [0, 180], 1)
        
        ret, track_window = cv2.meanShift(dest, track_window, term_criteria)
        
        x, y, w, h = track_window
        
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        
        cv2.imshow('Frame', img2)
        
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()

# %% [markdown]
# ## CAMShift Tracking 

# %%
cap = cv2.VideoCapture(0)

ret, frame = cap.read(0)

face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')

face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0])

track_window = (face_x, face_y, w, h)

roi = frame[face_y:face_y + h, face_x:face_x + w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_histogram = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

cv2.normalize(roi_histogram, roi_histogram, 0, 255, cv2.NORM_MINMAX)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read(0)
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        dest = cv2.calcBackProject([hsv], [0], roi_histogram, [0, 180], 1)
        
        ret, track_window = cv2.CamShift(dest, track_window, term_criteria)
        
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        
        img2 = cv2.polylines(frame, [pts], True, (0, 0, 255), 5)
        
        cv2.imshow('Frame', img2)
        
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
