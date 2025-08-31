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
# # Using Video Files

# %%
import cv2

# %% [markdown]
# ## Read a Video File

# %%
cap = cv2.VideoCapture('../DATA/finger_move.mp4')

if not cap.isOpened():
    print('Error: file not found or a wrong codec used')
    
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret == True:
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    else:
        break
        
cap.release()
cv2.destroyAllWindows()

# %% [markdown]
# ## Read a Video File (For Us to Watch)

# %%
import time

cap = cv2.VideoCapture('../DATA/finger_move.mp4')

if not cap.isOpened():
    print('Error: file not found or a wrong codec used')
    
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret == True:
        time.sleep(1/60)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    else:
        break
        
cap.release()
cv2.destroyAllWindows()
