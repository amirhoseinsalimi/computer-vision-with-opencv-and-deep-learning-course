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
# # Face Detection with OpenCV

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Loading Images

# %%
nadia = cv2.imread('../DATA/Nadia_Murad.jpg', 0)
denis = cv2.imread('../DATA/Denis_Mukwege.jpg', 0)
solvay = cv2.imread('../DATA/solvay_conference.jpg', 0)

# %%
plt.imshow(nadia, cmap='gray')

# %%
plt.imshow(denis, cmap='gray')

# %%
plt.imshow(solvay, cmap='gray')

# %% [markdown]
# ## Face Cascades

# %%
face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')

# %%
face_cascade


# %%
def detect_face(img):
    face_img = img.copy()
    
    face_rects = face_cascade.detectMultiScale(face_img)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return face_img


# %%
result = detect_face(nadia)
plt.imshow(result, cmap='gray')

# %%
result = detect_face(denis)
plt.imshow(result, cmap='gray')

# %%
result = detect_face(solvay)
plt.imshow(result, cmap='gray')


# %%
def adj_detect_face(img):
    face_img = img.copy()
    
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return face_img


# %%
result = adj_detect_face(solvay)
plt.imshow(result, cmap='gray')

# %% [markdown]
# ## Eye Cascades

# %%
eye_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')


# %%
def detect_eyes(img):
    face_img = img.copy()
    
    eyes_rects = eye_cascade.detectMultiScale(face_img)
    
    for (x, y, w, h) in eyes_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return face_img


# %%
result = detect_eyes(nadia)
plt.imshow(result, cmap='gray')

# %%
result = detect_eyes(denis)
plt.imshow(result, cmap='gray')

# %%
result = detect_eyes(solvay)
plt.imshow(result, cmap='gray')

# %% [markdown]
# ## Detecting the Face from a Video

# %%
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)
    
    frame = detect_face(frame)
    
    k = cv2.waitKey(1)
    
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
