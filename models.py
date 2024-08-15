from flask import Flask, request, redirect, url_for, render_template, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy


models = {
    'mango': tf.keras.models.load_model('models/mango_model.h5', compile=False),
    'banana': tf.keras.models.load_model('models/banana_model.h5', compile=False),
    'corn': tf.keras.models.load_model('models/corn_model.h5', compile=False),
    'cotton': tf.keras.models.load_model('models/cotton_model.h5', compile=False),
    'eggplant': tf.keras.models.load_model('models/eggplant_model.h5', compile=False),
    'peach': tf.keras.models.load_model('models/peach_model.h5', compile=False),
    'pepper': tf.keras.models.load_model('models/pepper_model.h5', compile=False),
    'mango': tf.keras.models.load_model('models/mango_model.h5', compile=False),
    'potato': tf.keras.models.load_model('models/potato_model.h5', compile=False),
    'rice': tf.keras.models.load_model('models/rice_model.h5', compile=False),
    'tomato': tf.keras.models.load_model('models/tomato_model.h5', compile=False),
    'grapes': tf.keras.models.load_model('models/grapes_model.h5', compile=False)
}

for key, model in models.items():
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Use string identifier for standard losses
        metrics=['accuracy']
    )




