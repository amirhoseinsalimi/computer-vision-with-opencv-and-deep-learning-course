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
# <a href="https://www.pieriandata.com"><img src="../DATA/Logo.jpg"></a>
# *Copyright by Pierian Data Inc.*

# %% [markdown]
# # Object Detection Assessment Project Exercise
#
# ## Russian License Plate Blurring
#
# Welcome to your object detection project! Your goal will be to use Haar Cascades to blur license plates detected in an image!
#
# Russians are famous for having some of the most entertaining DashCam footage on the internet (I encourage you to Google Search "Russian DashCam"). Unfortunately a lot of the footage contains license plates, perhaps we could help out and create a license plat blurring tool?
#
# OpenCV comes with a Russian license plate detector .xml file that we can use like we used the face detection files (unfortunately, it does not come with license detectors for other countries!)
#
# ----
#
#
# #### 3 Ways to Approach this project:
# * Just go for it! Use the image under the DATA folder called car_plate.jpg and create a function that will blur the image of its license plate. Check out the Haar Cascades folder for the correct pre-trained .xml file to use.
# * Use this notebook! Here we offer a guide of what main steps you should take to complete the project.
# * Jump to the solutions notebook and video to treat this entire project as code-along project where you can code along with us.
#
# ## Project Guide
#
# Follow and complete the tasks below to finish the project!

# %% [markdown]
# **TASK: Import the usual libraries you think you'll need.**

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# **TASK: Read in the car_plate.jpg file from the DATA folder.**

# %%
img = cv2.imread('../DATA/car_plate.jpg')

# %%
img.shape


# %% [markdown]
# **TASK: Create a function that displays the image in a larger scale and correct coloring for matplotlib.**

# %%
def display(img):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)


# %%
display(img)

# %% [markdown]
# **TASK: Load the haarcascade_russian_plate_number.xml file.**

# %%
plate_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_russian_plate_number.xml')


# %% [markdown]
# **TASK: Create a function that takes in an image and draws a rectangle around what it detects to be a license plate. Keep in mind we're just drawing a rectangle around it for now, later on we'll adjust this function to blur. You may want to play with the scaleFactor and minNeighbor numbers to get good results.**

# %%
def detect_plate(img):
    plate_img = img.copy()
    
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=3) 
    
    for (x, y, w, h) in plate_rects:
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (255, 0, 0), 4)
        
    return plate_img


# %%
result = detect_plate(img)

# %%
display(result)


# %% [markdown]
# **FINAL TASK: Edit the function so that is effectively blurs the detected plate, instead of just drawing a rectangle around it. Here are the steps you might want to take:**
#
# 1. The hardest part is converting the (x,y,w,h) information into the dimension values you need to grab an ROI (somethign we covered in the lecture 01-Blending-and-Pasting-Images. It's simply [Numpy Slicing](https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python), you just need to convert the information about the top left corner of the rectangle and width and height, into indexing position values.
# 2. Once you've grabbed the ROI using the (x,y,w,h) values returned, you'll want to blur that ROI. You can use cv2.medianBlur for this.
# 3. Now that you have a blurred version of the ROI (the license plate) you will want to paste this blurred image back on to the original image at the same original location. Simply using Numpy indexing and slicing to reassign that area of the original image to the blurred roi.

# %%
def detect_and_blur_plate(img):
    plate_img = img.copy()
    
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=3) 
    
    for (x, y, w, h) in plate_rects:
        blured_plate = cv2.medianBlur(plate_img[y:y + h, x:x + w], 7)
        
        plate_img[y:y + h, x:x + w] = blured_plate
        
        return plate_img


# %%
result = detect_and_blur_plate(img)

# %%
display(result)

# %% [markdown]
# # Great Job!
