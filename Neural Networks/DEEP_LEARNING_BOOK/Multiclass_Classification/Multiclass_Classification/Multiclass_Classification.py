
import keras
from keras import models
from keras import layers
from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
import copy

import matplotlib.pyplot as plt


#print(keras.__version__)
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#print(len(train_data))
#print(len(test_data))

#print(train_data[10])


##decoding algorithm
#word_index = reuters.get_word_index()
#reverse_word_index = dict([(value, key) for (key, value) in
#word_index.items()])
## Kod dekodujący recenzję.  Zauważ, że indeksy są przesunięte o 3, ponieważ
## pod indeksami o numerach 0, 1 i 2
## znajdują się indeksy symbolizujące „wypełnienie”, „początek sekwencji” i
## „nieznane słowo”.
#decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in
#train_data[0]])

#print(decoded_newswire)
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Zbiór treningowy w postaci wektora.
x_train = vectorize_sequences(train_data)
# Zbiór testowy w postaci wektora.
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

## Etykiety zbioru treningowego w postaci wektora.
#one_hot_train_labels = to_one_hot(train_labels)
## Etykiety zbioru testowego w postaci wektora.
#one_hot_test_labels = to_one_hot(test_labels)
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)



model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20,batch_size=512, validation_data=(x_val, y_val))


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.show()


plt.clf()   # Czyszczenie rysunku.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.show()


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)

print(results)


test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)

print(float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels))


predictions = model.predict(x_test)

print(predictions.shape)
print(predictions[0].shape)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))