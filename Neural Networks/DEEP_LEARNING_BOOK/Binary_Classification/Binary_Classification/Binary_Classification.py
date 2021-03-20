import tensorflow as tf
import numpy as np
from keras.datasets import imdb
import keras
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt


#The variables train_data and test_data are lists of reviews, each review being a list of word indices (encoding a sequence of words). train_labels and test_labels are lists of 0s and 1s, where 0 stands for "negative" and 1 stands for "positive" 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #num_words=10000 -> means that we will only get the top of 10000 most frequently occurred words

print(train_data[0])



# word_index is a dictionary mapping words to an integer index
#word_index = imdb.get_word_index()
## We reverse it, mapping integer indices to words
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
## We decode the review; note that our indices were offset by 3
## because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
#decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

#print(decoded_review)

print(len(train_data[0]))

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

print(x_train[0])

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#              loss=losses.binary_crossentropy,
#              metrics=[metrics.binary_accuracy])


x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))



history_dict = history.history
print(history_dict.keys())
#print(history_dict.keys())


#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']

#epochs = range(1, len(acc) + 1)

## Parametr bo definiuje linię przerywaną w postaci niebieskich kropek.
#plt.plot(epochs, loss, 'bo', label='Strata trenowania')
## Parametr b definiuje ciągłą niebieską linię.
#plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
#plt.title('Strata trenowania i walidacji')
#plt.xlabel('Epoki')
#plt.ylabel('Strata')
#plt.legend()

#plt.show()


#plt.clf()   # Czyszczenie rysunku.
#acc_values = history_dict['accuracy']
#val_acc_values = history_dict['val_accuracy']

#plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
#plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
#plt.title('Dokladnosc trenowania i walidacji')
#plt.xlabel('Epoki')
#plt.ylabel('Strata')
#plt.legend()

#plt.show()

results = model.evaluate(x_test, y_test)
print(results)

print(model.predict(x_test))