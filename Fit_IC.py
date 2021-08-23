from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
import numpy as np
import os
from tqdm import tqdm
import string
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


os.chdir("C:\\Users\\stat\\Desktop\\DJ_project\\tts\\IC") # model 저장된 경로

# 캡셔닝 모델 생성
def LSTM_model(vocab_size, max_length):
    inputs1 = Input(shape=(1000,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = Dropout(0.5)(fe2)
    fe4 = Dense(128, activation='relu')(fe3)

    # 시퀀스 모델
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    # mask_zero=True : 0으로 패딩된 값을 마스킹하여 네트워크의 뒤로 전달되지 않게 만든다.
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, return_sequences=True)(se2)
    se4 = Dropout(0.5)(se3)
    se5 = LSTM(128, return_sequences=False)(se4)

    # 디코더
    decoder1 = add([fe4, se5])
    decoder2 = Dense(128, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # [image, seq] [word]로 묶기
    caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # caption_model.summary()
    return caption_model


def real_extract_features(filename):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))

    # load Image
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)  # numpy array로 변환
    image = np.expand_dims(image, axis=0)

    # 특징 추출
    feature = model.predict(image, verbose=0)
    feature = np.reshape(feature, OUTPUT_DIM)

    return feature


OUTPUT_DIM = 1000
embedding_dim = 200
max_length = 34

START = "startseq"
STOP = "endseq"

wordtoidx = load(open('wti.pkl', 'rb'))
idxtoword = load(open('itw.pkl', 'rb'))

vocab_size = len(idxtoword) + 1

embedding_matrix = np.loadtxt('embedded.txt')
embeddings_index = load(open('ebd_i.pkl', 'rb'))

for word, i in wordtoidx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


caption_model = LSTM_model(vocab_size, max_length)
caption_model.layers[2].set_weights([embedding_matrix])
caption_model.layers[2].trainable = False

# 모델 컴파일
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

model_path = os.path.join(os.getcwd(), f'caption_model.h5')
caption_model.load_weights(model_path)

def generateCaption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final



class ImgC:
    def __init__(self, image):
        self.imagePath = image

    def fit(self):

        file = 'img.jpg'
        photo = real_extract_features(file)
        photo = photo.reshape((1, OUTPUT_DIM))

        # 설명 생성
        description = generateCaption(photo)
        print(description)

        return description
