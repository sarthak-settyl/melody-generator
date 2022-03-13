from glob import glob
from re import S
from turtle import color
import tensorflow as tf
from model.generator import define_generator
import argparse
from colored import fg
from datetime import datetime
from utils.image2midi import image2midi
import time

parser = argparse.ArgumentParser(description="Arguments for the melody generation")
parser.add_argument('--instrument',metavar='-i',help='Instrument for which audio output will be generated',default='drum')
parser.add_argument("--sequence",metavar='-s',help='Single duration dpicts ~7 second of playtime',default=1,type=int)
args = parser.parse_args()

def instrument(argument):
    switcher = {
        'drum':'weights/final_drum_generator_weights.h5'
        }
    try:
        return switcher.get(argument, "weights/final_drum_generator_weights.h5")

    except:
        color = fg('red')
        print(color+f'Erro with the instrument current availabe instrument are {switcher}')

playing_instrument_path = instrument(args.instrument)

model = None
IMG_SHAPE = (106, 106, 1)
BATCH_SIZE = 64

# Size of the noise vector
noise_dim = 128

def get_model():
    color = fg('blue')
    print(color+f'Loading model')
    global model
    model = define_generator(noise_dim)
    return model
get_model()
model.load_weights(playing_instrument_path)
color = fg('green')
print(color+f'model loaded')

try:
    color = fg('blue')
    print(color+f'Generating melody')
    seed = tf.random.normal([args.sequence,128])
    prediction = model(seed,training=False)
    output = []
    for data in prediction:
        data = (data*127.5) + 127.5
        output.append(data)
    data = tf.concat(output,axis=1)
    img = tf.keras.preprocessing.image.array_to_img(data)
    now = datetime.now()
    img.save(f'audio.png')
    time.sleep(3)
    image2midi(f'audio.png')
    color = fg('green')
    print(color+f'melody generated with name :- {now}.midi')
except Exception as e:
    print(f"error {e}")