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
#     version: 3.6.10
# ---

# %% [markdown]
# # Deep Learning on Custom Images - Part 2

# %% [markdown]
# ## Import Libraries

# %%
import cv2
from keras.preprocessing.image import ImageDataGenerator, image
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Inspect the Data

# %%
cat4 = cv2.imread('../DATA/CATS_DOGS/train/CAT/4.jpg')

cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)

plt.imshow(cat4);

# %%
cat4.shape

# %%
dog2 = cv2.imread('../DATA/CATS_DOGS/train/DOG/2.jpg')

dog2 = cv2.cvtColor(dog2, cv2.COLOR_BGR2RGB)

plt.imshow(dog2);

# %%
dog2.shape

# %% [markdown]
# Note that the shape of the images is different.

# %% [markdown]
# ## Image Data Generator

# %%
dog2.max()

# %%
image_generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# %%
plt.imshow(image_generator.random_transform(dog2))

# %%
image_generator.flow_from_directory('../DATA/CATS_DOGS/train/')

# %% [markdown]
# ## Building the Model

# %%
input_shape=(150, 150, 3)

# %%
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
model.summary()

# %% [markdown]
# ### Preparing the Data

# %%
batch_size = 16

train_image_generator = image_generator.flow_from_directory(
    '../DATA/CATS_DOGS/train',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# %%
test_image_generator = image_generator.flow_from_directory(
    '../DATA/CATS_DOGS/test',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# %%
train_image_generator.class_indices

# %%
results = model.fit_generator(
    train_image_generator,
    epochs=1,
    steps_per_epoch=150,
    validation_data=test_image_generator,
    validation_steps=12
)

# %%
results.history['acc']

# %%
results.history

# %% [markdown]
# ## Loading a Pre-trained Model

# %%
new_model = load_model('../DATA/cat_dog_100epochs.h5')

# %%
dog_file = '../DATA/CATS_DOGS/test/DOG/10005.jpg'

dog_image = image.load_img(dog_file, target_size=input_shape[:2])

dog_image = image.img_to_array(dog_image)

print(dog_image.shape)

dog_image = np.expand_dims(dog_image, axis=0)

dog_image = dog_image / 255

# %%
new_model.predict_classes(dog_image)

# %%
new_model.predict(dog_image)
