#!/usr/bin/env python
# coding: utf-8

# spli_folders.py
"""
Installtion the data-splitting library
Please install split_folders with the pip

$ pip install split-folders

Usage: 

It is a typical way to splot the dataset to the three sets including train, val and test
set with a given ratio such as 0.8, 0.1 and 0.1.  To only split into training and val set, 
please set a tuple to `ratio`, i.e, `(.8, .2)`.

Split val/test with a fixed number of items e.g. 100 for each set.
To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
splitfolders.fixed("input_folder", output="output", seed=1337, fixed=(100, 100), oversample=False, group_prefix=None)

Please use the original dataset name or the new name that you prefer to. Here use the new 
name of furniture_pictures for the downloaded dataset. 
"""


import splitfolders


# The path to the directory where the original dataset was uncompressed. 
input_folder = '/home/mike/datasets/furniture_pictures'

# The directory store the smaller dataset 
output_folder = '/home/mike/Documents/keras_incpetion_v3/Visual_Search/furniture_pictures'

# Get the train, val and test sets. 
splitfolders.ratio(input_folder, output_folder, seed=1337, ratio=(.8, .1, .1), group_prefix=None) 