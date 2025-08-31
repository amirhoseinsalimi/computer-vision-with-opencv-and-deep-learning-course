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
# # Drawing on Live Camera

# %%
import cv2

# %% [markdown]
# ## Draw Shapes on Videos

# %% [markdown]
# ### Static Position

# %%
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

x = width // 2
y = height // 2

w = width // 4
h = height // 4

while True:
    ret, frame = cap.read()
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# %% [markdown]
# ### Interactive Position

# %%
def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, top_left_clicked, bottom_right_clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if top_left_clicked and bottom_right_clicked:
            pt1 = (0, 0)
            pt2 = (0, 0)
            top_left_clicked = False
            bottom_right_clicked = False
    
        if not top_left_clicked:
            pt1 = (x, y)
            top_left_clicked = True
        elif not bottom_right_clicked:
            pt2 = (x, y)
            bottom_right_clicked = True
            

pt1 = (0, 0)
pt2 = (0, 0)
top_left_clicked = False
bottom_right_clicked = False

cap = cv2.VideoCapture(0)
cv2.namedWindow('Test')
cv2.setMouseCallback('Test', draw_rectangle)

while True:
    ret, frame = cap.read()
    
    if top_left_clicked:
        cv2.circle(frame, center=pt1, radius=5, color=(0, 0, 255), thickness=-1)
        
    if top_left_clicked and bottom_right_clicked:
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 3)
    
    cv2.imshow('Test', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
