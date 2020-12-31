import os
from keras.preprocessing.image import ImageDataGenerator

rootPath = "./datasets/cat-and-dog"

imageGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=.1
)

trainGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'training_set'),
    target_size=(64, 64),
    subset='training'
)

validationGen = imageGenerator.flow_from_directory(
    #os.path.join(rootPath, 'test_set'),
    os.path.join(rootPath, 'training_set'),
    target_size=(64, 64),
    subset='validation'
)

#모델생성
from keras.models import Sequential
from keras import layers

model = Sequential()

model.add(layers.InputLayer(input_shape=(64, 64, 3)))   #이미지 크기 64x64 , 색상을 갖고있기때문에 3(red,green blue)
model.add(layers.Conv2D(16, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc'],
)

#체크포인트 콜백 사용
import tensorflow as tf
from keras.callbacks import  EarlyStopping

model_dir = './log'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + "/dog_cat3.model"

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)


history=model.fit_generator(
    trainGen,
    epochs=32,
    steps_per_epoch=trainGen.samples/32,
    validation_data=validationGen,
    validation_steps=trainGen.samples/32,
    callbacks=[checkpoint,early_stopping]
)

model.save_weights('dog-cat-classifier2.h5')
