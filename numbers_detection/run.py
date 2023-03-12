import keras
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import plot_model
# from keras.callbacks import LambdaCallback, ModelCheckpoint

EPOCHS: int = 2
BATCH_SIZE: int = 32
LOAD_MODEL: bool = False
MODEL_FILENAME: str = 'numbers_detection.h5'

print()

# GET DATASETS


def get_numbers_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    return mnist.load_data()


(train_dataset, train_dataset_marks), (test_dataset, test_dataset_marks) = get_numbers_data()

print(f'{len(train_dataset)=}')
print(f'{len(train_dataset_marks)=}')
print(f'{len(test_dataset)=}')
print(f'{len(test_dataset_marks)=}')

# SHOW EXAMPLES
# def show_example_data(train_data: list[np.ndarray], columns: int = 4, rows: int = 5) -> None:
#     figure = plt.figure(
#         # figsize=(8, 8)
#     )

#     for i, img in enumerate(train_data[:4*5], start=1):
#         # print(f"shape of {i} image => {img.shape[1:]}")
#         figure.add_subplot(rows, columns, i)
#         plt.imshow(img)


# show_example_data(train_dataset)

# PREPARE DATASET

def prepare_data(raw_data: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
    ret = raw_data.astype("float32") / 255
    ret = ret.reshape(-1, 28*28)
    # for i, img in enumerate(raw_data):
    #     img = img.astype(np.float32) / 255.0
    #     img = img.reshape((img.shape[0]*img.shape[1], )) # (784, ) shape
    #     ret[i] = img

    return ret


train_prepared_dataset = prepare_data(train_dataset)
test_prepared_dataset = prepare_data(test_dataset)
train_prepared_dataset_marks = keras.utils.to_categorical(train_dataset_marks, 10)
test_prepared_dataset_marks = keras.utils.to_categorical(test_dataset_marks, 10)
input_shape: tuple[int] = train_prepared_dataset[0].shape

print(f'{len(train_prepared_dataset)=}, {len(test_prepared_dataset)=}')
print(f'img with number => {train_dataset_marks[0]} with size {train_prepared_dataset[0].size} and {train_prepared_dataset[0].shape} linear shape')  # noqa
print(f'{input_shape=}')


# MODEL

model = keras.models.load_model(MODEL_FILENAME) if LOAD_MODEL is True else Sequential([
    Input(shape=input_shape),
    Dense(800, activation='relu'),
    Dense(200, activation='relu'),
    Dense(10, activation='softmax')
])

print(f'model successfully created => {model}')

# Ошибка:sparse_categorical_crossentropy
# оптимизатор:adam
# метрика:accuracy

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
plot_model(model, dpi=120)

# TRAIN

if not LOAD_MODEL:
    # with tf.device('/device:GPU:0'):
    history = model.fit(train_prepared_dataset, train_prepared_dataset_marks, BATCH_SIZE, EPOCHS, validation_split=0.1)  # noqa
    print(history)
    model.save('numbers_detection.h5')
else:
    print('GET MODEL FROM FILE')
    model = keras.models.load_model('numbers_detection.h5')


def test(test_index: int, show_img: bool = False) -> None:

    img_arr = test_prepared_dataset[test_index]
    value = test_prepared_dataset_marks[test_index]

    correct_ans = np.argmax(value)
    img_expanded = np.expand_dims(img_arr, axis=0)

    if show_img:
        plt.imshow(img_arr.reshape(28, 28))

    result = model.predict(img_expanded)
    # print('result => ' + str(result))
    predicted_number = np.argmax(result[0])
    # if correct_ans != predicted_number:

    print(f"LOOKS LIKE THIS IS \"{predicted_number}\" NUMBER WITH \"{result[0][predicted_number]:.2f}\"%")
    print(f'IS ANSWER CORRECTNES => {correct_ans == predicted_number}, index => {test_index}')


test(35)
