import os
import json
import argparse
import subprocess
from PIL import Image
import numpy as np
import pandas as pd
import collections
from zipfile import ZipFile
import boto3
# from tqdm import tqdm

import MeCab

from tensorflow.keras.models              import Model
from tensorflow.keras.applications.vgg16  import VGG16 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import reshape, stack, convert_to_tensor

# bokekan = ['blue_000', 'yellow_000', 'yellow_001', 'yellow_002', 'yellow_003','yellow_004', 'yellow_005', 'yellow_006','yellow_007', 'yellow_008', 'yellow_009', 'green_000', 'red_000', 'sp_000']

processing_input = '/opt/ml/processing/input' # zipped file
processing_bokekan_contents = '/opt/ml/processing/output/bokekan/contents' # intermediate data (raw data)
processing_bokekan_metadata = '/opt/ml/processing/output/bokekan/metadata' # Pandas DF -> CSV
processing_output = '/opt/ml/processing/output/training' # feature vector for model input

def load_data(bokekan):
    """Load Bokekan dataset.

    Keyword arguments:
    bokekan -- the list of bokekan datasets. e.g. ['green_000', 'red_000', 'sp_000']
    """
    df_list = []
    for boke_class in bokekan:
        file_path = os.path.join(processing_input, boke_class+'.zip')

        try:
            zipfile = ZipFile(file_path)
            print('Extracting...', file_path)
            zipfile.extractall(path=os.path.join(processing_bokekan_contents, boke_class))

        except:
            pass
        
        df_ = pd.read_csv(os.path.join(processing_bokekan_contents, boke_class, 'boke.csv'), lineterminator='\n')
        df_['bokekan_id'] = boke_class
        df_list.append(df_)
        
    return pd.concat(df_list, axis=0, ignore_index=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bokekan", 
#                         action="extend", # >= py38
                        nargs="+", type=str)
    args, _ = parser.parse_known_args()

    print('---- Load data ----')
        
    data = load_data(args.bokekan)
    data.to_csv(os.path.join(processing_bokekan_metadata, 'boke.csv'), index=False)
    
    print('---- Text preprocessing ----')
    
    # delete unnecessary characters
    data["text"] = data.text.str.replace("\n", "")
    
    # word tokenization
    subprocess.run(['python', '-m', 'unidic', 'download'])
    
    tagger = MeCab.Tagger('-Owakati')

    w_ys = []

    max_word = 0
    # put Beginning Of Sentence (BOS) and End Of Sentence (EOS)
    for txt in data['text']:
        ys = ["BOS"]
        ys.extend(tagger.parse(txt).replace("\n", "").split())
        ys.append("EOS")
        w_ys.append(ys)

        # check the max word length
        if max_word < len(ys):
            max_word = len(ys)
    
    # create the index of words
    corpus = []
    for word in w_ys:
        corpus.extend(word)

    # create list of tuples in descending order of frequency
    # collections.Counter(corpus).items(): Dictionay {key: word, value: frequency} -> [(key, val), (key, val), ...]
    # key=lambda x: x[1]: `sorted` makes `val` sort key
    word_id = [(k, v) for k, v in sorted(collections.Counter(corpus).items(), key=lambda x: x[1], reverse=True) if k!="EOS"]

    # Create Dictionary {Word: ID}, where ID >= 1
    word_id = { k: e for e, (k, v) in enumerate(word_id, start = 1)}
    word_id["EOS"] = 0

    # Create a reverse Dictionary {ID: Word} 
    id_word = { v: k for k, v in word_id.items()}
    
    print('---- Image preprocessing ----')
    
    # image preprocessing (embedding using VGG16)
    VGG = VGG16(weights='imagenet')
    VGG._layers.pop()
    VGG = Model(inputs=VGG.inputs, outputs=VGG.layers[-1].output)

    # assign 1 text data (list of words) per 1 image
    X1 = [] # image vector output embedded by VGG16 
    X2 = [] # text vector
    y = []  # predict next word 

    for i,(boke,url,directory) in enumerate(zip(w_ys, data.odai_photo_url, data.bokekan_id)):
        try:
            # 画像ロード
            path = os.path.join(processing_bokekan_contents, directory)
            img = load_img(path+url, target_size=(224, 224))
            x = np.array(img, dtype=np.float32)/255
            x = VGG(reshape(x, [1, 224, 224, 3])) 

        except:
            print("Cannot load", path+url)
            continue

        # 0-padding. set the next word to predict. 
        for i2,key in enumerate(boke):
            if key != "BOS":
                y.append(word_id[key])
            if key == "EOS":
                break

            x2k = []
            X1.append(reshape(x, [-1])) 
            word = boke[:i2+1]
            x2k.extend([0 for n in range(max_word - len(word))])
            for w in word:
                x2k.append(word_id[w])
            X2.append(x2k)

    X1 = stack(X1)
    X2 = stack(X2)
    
    print('---- Save data ----')
    
    vocabulary_size = max(map(int, id_word.keys()))+1

    # save metadata for training/inference, dictionary for inference
    with open(os.path.join(processing_bokekan_metadata, 'param.json'), 'w') as f:
        json.dump({'max_word': max_word, 'vocabulary_size': vocabulary_size}, f)
    pd.DataFrame(word_id.items()).to_csv(os.path.join(processing_bokekan_metadata, 'word_id_bokete_data.csv'), index=False)
    pd.DataFrame(id_word.items()).to_csv(os.path.join(processing_bokekan_metadata, 'id_word_bokete_data.csv'), index=False)
    
    # save training data
    np.savez(os.path.join(processing_output, 'X1.npz'), X1.numpy())
    np.savez(os.path.join(processing_output, 'X2.npz'), X2.numpy())
    np.savez(os.path.join(processing_output, 'y.npz'), np.array(y, dtype=np.int))
    
    print('---- Finished ----')