
import keras
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

print(keras.__version__)

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)


#Perparing data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    # Będziemy tworzyć wiele instancji tego samego modelu,
    # a więc konstruując je, będziemy korzystać z funkcji.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
#for i in range(k):
#    print('processing fold #', i)
#    # Przygotuj dane walidacyjne: dane z k-tej składowej.
#    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

#    # Przygotuj dane treningowe: dane z pozostałych składowych.
#    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
#         train_data[(i + 1) * num_val_samples:]],
#        axis=0)
#    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
#         train_targets[(i + 1) * num_val_samples:]],
#        axis=0)

#    # Zbuduj model Keras (model został skompilowany wcześniej).
#    model = build_model()
#    # Trenuj model w trybie cichym (parametr verbose = 0).
#    model.fit(partial_train_data, partial_train_targets,
#              epochs=num_epochs, batch_size=1, verbose=0)
#    # Przeprowadź ewaluację modelu przy użyciu danych walidacyjnych.
#    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#    all_scores.append(val_mae)


print(all_scores)
print(np.mean(all_scores))

# Some memory clean-up
K.clear_session()


#num_epochs = 500
#all_mae_histories = []
#for i in range(k):
#    print('processing fold #', i)
#    # Przygotowuje dane walidacyjne: dane z k-tej składowej.
#    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

#    # Przygotowuje dane treningowe: dane z pozostałych składowych.
#    partial_train_data = np.concatenate(
#        [train_data[:i * num_val_samples],
#         train_data[(i + 1) * num_val_samples:]],
#        axis=0)
#    partial_train_targets = np.concatenate(
#        [train_targets[:i * num_val_samples],
#         train_targets[(i + 1) * num_val_samples:]],
#        axis=0)

#    # Buduje model Keras (model został skompilowany wcześniej).
#    model = build_model()
#    # Przeprowadza ewaluację modelu przy użyciu danych walidacyjnych.
#    history = model.fit(partial_train_data, partial_train_targets,
#                        validation_data=(val_data, val_targets),
#                        epochs=num_epochs, batch_size=1, verbose=0)
#    mae_history = history.history['val_mean_absolute_error']
#    all_mae_histories.append(mae_history)


#average_mae_history = [
#    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


#plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
#plt.xlabel('Liczba epok')
#plt.ylabel('Sredni blad bezwzgledny')
#plt.show()

# Utwórz nową, skompilowaną wersję modelu.
model = build_model()
# Trenuj model na całym zbiorze danych treningowych.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


print(test_mae_score)