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
# # NumPy Arrays

# %% [markdown]
# ## Image Creation Methods

# %%
import numpy as np

# %%
mylist = [1, 2, 3]

# %%
type(mylist)

# %%
myarray = np.array(mylist)

# %%
myarray

# %%
type(myarray)

# %%
np.arange(0, 10)

# %%
np.arange(0, 10, 2)

# %%
np.zeros(shape=(5, 5))

# %%
np.zeros(shape=(10, 5))

# %%
type(0)

# %%
type(0.)

# %%
type(.0)

# %%
np.ones(shape=(2, 4))

# %% [markdown]
# ## Operations

# %%
np.random.seed(101)

arr = np.random.randint(0, 100, 10)

# %%
arr

# %%
arr2 = np.random.randint(0, 100, 10)

# %%
arr2

# %%
arr.max()

# %%
arr.argmax()

# %%
arr.min()

# %%
arr.argmin()

# %%
arr.mean()

# %%
arr.shape

# %%
arr.reshape((5, 2))

# %% [markdown]
# ## Indexing & Slicing

# %%
mat = np.arange(0, 100).reshape(10, 10)

# %%
mat

# %%
mat.shape

# %%
row = 0
col = 1

# %%
mat[row][col]

# %%
mat[4, 6]

# %%
mat[:, 1]

# %%
mat[:, 1].shape

# %%
mat[:, 1].reshape(10, 1)

# %%
mat[2, :]

# %%
mat[0:3, 0:4]

# %%
mat[0:3, 0:4] = 0

# %%
mat

# %%
mynewmat = mat.copy()

# %%
mynewmat[0:6, :] = 999

# %%
mynewmat

# %%
mat
