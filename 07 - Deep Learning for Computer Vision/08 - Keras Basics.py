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
# # Keras Basics

# %% [markdown]
# ## Data Preprocessing

# %%
import numpy as np 
from numpy import genfromtxt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model

# %%
data = genfromtxt('../DATA/bank_note_data.txt', delimiter=',')

# %%
data

# %%
labels = data[:, -1]

# %%
labels

# %%
features = data[:, 0:4]

# %%
features

# %%
X = features

y = labels

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
X_train

# %%
len(X_train)

# %%
X

# %%
len(X)

# %%
len(X_test)

# %%
X_test

# %%
y_train

# %%
y_test

# %%
scaler_object = MinMaxScaler() 

# %%
X_train.min(), X_train.argmin(), X_train.max(), X_train.argmax()

# %%
scaler_object.fit(X_train)

# %%
scaled_X_train = scaler_object.transform(X_train)

# %%
scaled_X_test = scaler_object.transform(X_test)

# %%
scaled_X_train.min(), scaled_X_train.argmin(), scaled_X_train.max(), scaled_X_train.argmax()

# %%
X_train

# %%
scaled_X_train

# %% [markdown]
# ## Building the Model

# %%
from keras.models import Sequential
from keras.layers import Dense

# %%
model = Sequential()

model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)

# %%
predictions = model.predict_classes(scaled_X_test)

predictions[:10]

# %%
model.metrics_names

# %%
confusion_matrix(y_test, predictions)

# %%
print(classification_report(y_test, predictions))

# %%
model.save('my_model.h5')

# %%
new_model = load_model('my_model.h5')

# %%
new_model.predict_classes(scaled_X_test)[:10]
