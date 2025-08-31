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
# # Keras Convolutional Neural Networks with MNIST

# %% [markdown]
# ## Import Libraries

# %%
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ## Load the Data

# %%
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# %% [markdown]
# ### Inspect the Data

# %%
X_train.shape

# %%
single_image = X_train[0]

single_image

# %%
plt.imshow(single_image, cmap='gray_r')

# %%
y_train

# %%
y_train.shape

# %% [markdown]
# ## Preproces the Data

# %% [markdown]
# ### One-hot Encoding

# %%
y_categorical_train = to_categorical(y_train, 10)
y_categorical_test = to_categorical(y_test, 10)

# %%
y_categorical_train[:3]

# %%
single_image.max()

# %% [markdown]
# ### Scale the Data

# %%
X_train = X_train / X_train.max()

# %%
X_test = X_test / X_test.max()

# %%
scaled_image = X_train[0]

scaled_image.max(), scaled_image.min(), scaled_image.mean()

# %%
plt.imshow(scaled_image, cmap='gray_r')

# %% [markdown]
# ### Reshape the Data

# %%
X_train.shape

# %%
X_train = X_train.reshape((60000, 28, 28, 1))

# %%
X_test = X_test.reshape((10000, 28, 28, 1))

# %%
X_train.shape

# %%
plt.imshow(scaled_image, cmap='gray_r')

# %% [markdown]
# ## Building the Model

# %%
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

# %% [markdown]
# ### Train and Evaluate the Model

# %%
model.fit(X_train, y_categorical_train, epochs=2)

# %%
model.metrics_names

# %%
model.evaluate(X_test, y_categorical_test)

# %%
predictions = model.predict_classes(X_test)

# %%
y_categorical_test

# %%
predictions

# %%
y_test

# %%
print(classification_report(y_test, predictions))
