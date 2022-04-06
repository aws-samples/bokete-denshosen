# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     https://aws.amazon.com/apache-2-0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import json
import logging
import os
import re

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, 
    LSTM, 
    Input, 
    Dropout, 
    Embedding, 
    add, 
    concatenate, 
    GlobalMaxPooling1D, 
    Flatten, 
    Bidirectional, 
    Attention, 
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import reshape, stack, convert_to_tensor
from tensorflow.io import write_file

import numpy as np
import pandas as pd

from keras_self_attention import SeqSelfAttention

logging.getLogger().setLevel(logging.INFO)

def main(args):
    # load data
    X1 = np.load(os.path.join(args.training, 'X1.npz'))['arr_0']
    X2 = np.load(os.path.join(args.training, 'X2.npz'))['arr_0']
    y = np.load(os.path.join(args.training, 'y.npz'))['arr_0']
    max_word = args.max_word
    vocabulary_size = args.vocabulary_size
    y = np.eye(vocabulary_size, dtype=np.float32)[y]
    
    X1 = convert_to_tensor(X1, dtype=tf.float32)
    X2 = convert_to_tensor(X2, dtype=tf.float32)
    y = convert_to_tensor(y, dtype=tf.int8)
    
    # define the model
    inputs1 = Input(shape=(4096,), name='image') 
    ie1 = Dense(512, activation='relu')(inputs1)
    ie2 = Dropout(0.5)(ie1)

    # text encoder
    inputs2 = Input(shape=(max_word,), name='text')
    se1 = Embedding(vocabulary_size, 256)(inputs2)
    se2 = Bidirectional(LSTM(256,return_sequences=True))(se1)
    se2 = SeqSelfAttention(name='attention')(se2)
    se3 = GlobalMaxPooling1D()(se2)
    se3 = Dropout(0.5)(se3)

    # decoder
    decoder1 = add([ie2, se3])
    decoder2 = Dense(1024, activation='relu')(decoder1)
    outputs = Dense(vocabulary_size, activation='softmax')(decoder2)

    # captioning 
    model = Model(inputs=[inputs1, inputs2], outputs=outputs, name='boke_base')
    model.compile(optimizer=args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_name = "weights.{epoch:02d}-{loss:.2f}.hdf5"
    
    model.summary()

    model_checkpoint = ModelCheckpoint(os.path.join(args.output_dir, model_name), monitor='loss', verbose=1, save_best_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='loss', patience=8, verbose=1, mode='auto')
    
    history = model.fit(
        {'image_input': X1, 'text_input': X2}, 
        y, 
        shuffle=True, 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        callbacks=[model_checkpoint, early_stopping], 
    )
    
    model.save(os.path.join(args.model_output_dir, '00000001')) # local -> S3 (.tar.gz)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_TRAINING"),
        help="The directory where the bokete input data is stored.",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="The directory where the model will be stored."
    )
    parser.add_argument("--model_output_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))                        
    parser.add_argument("--checkpoint-dir",type=str, default="/opt/ml/checkpoints", help="The directory where checkpoints will be saved.")
    parser.add_argument("--tensorboard-dir", type=str, default=os.environ.get("SM_MODULE_DIR"))
    parser.add_argument(
        "--weight-decay", type=float, default=2e-4, help="Weight decay for convolutions."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """,
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="The number of steps to use for training."
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument(
        "--data-config", type=json.loads, default=os.environ.get("SM_INPUT_DATA_CONFIG")
    )
    parser.add_argument(
        "--fw-params", type=json.loads, default=os.environ.get("SM_FRAMEWORK_PARAMS")
    )
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default="0.9")
    parser.add_argument("--max_word", type=int, default="49")
    parser.add_argument("--vocabulary_size", type=int, default="15240")
    args = parser.parse_args()
    main(args)