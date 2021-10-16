# Dane pochodzą ze zbioru
# https://www.kaggle.com/moltean/fruits
#
# Importy
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


# zmieniamy rozmiar obrazów
IMAGE_SIZE = [100, 100]

# hiperparametry sieci
epochs = 3
batch_size = 32

train_path = 'data/fruits-360/Training'
valid_path = 'data/fruits-360/Test'
# train_path = 'data/fruits-360-small/Training'
# valid_path = 'data/fruits-360-small/Test'

# Przyda się do określenia liczy obrazów
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# Przyad się do określenia liczby klas
folders = glob(train_path + '/*')


# Obejrzyjmy jeden obraz
plt.imshow(
  image.img_to_array(
    image.load_img(
      np.random.choice(image_files)
    )
  ).astype('uint8')
)
plt.show()


# Tworzymy model VGG16, który automatycznie załadowuje wytrenowane wagi
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# Parametry:
# input_shape - musimy wskazać rozmiar wejściowy dany
# weights - którego zbioru wytrenowanych wag chcemy użyć (domyślny - imagenet)
# include_top - chcemy wszystko poza ostatnią warstwą, która pełni rolę klasyfikatora

# Nie trenujemy żadnej istniejących wag modelu VVG
for layer in vgg.layers:
  layer.trainable = False

# Tworzymy nasze warstwy (klasyfikator)
# Można więcej jeżeli chcemy
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)
# softmax (bo multiclassification)

# inicjalizujemy model
model = Model(inputs=vgg.input, outputs=prediction)

# Wyświetlamy charakterystyki modelu
print(model.summary())

# wskazujemy funkcje kosztu, optymizer i metryki, które chcemy wyświetlić
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)


# inicjalizujemy ImageDataGenerator (Data Augmentation)
# Model będzie mocniej generalizował
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input  # VGG jast napisane w bibliotece
  # caffe, caffe przechowuje obrazy w przestrzeni BGR, a nie RGB i sieć
  # jest wytrenowana na takie dane. Dlatego ta funkcja. Zrealizuje
  # potrzebne rzutowanie. Dodatkowo odejmie średnią od wszystkich piksel,
  # żeby średnia była równa 0
)

# Przetestujmy generator
# obsługa mapowania etykiet (żeby było wiadomo
# jaką etykiete ma wygenerowany obraz)
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE) # inicjalizujemy
# generator (testowy)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k

# No to zobaczmy jak wygląda wygenerowane zdjęcie.
# Jeżeli kolory nie będą dziwne oznaczać to, że
# kolejność kanałów z powodu jakiegoś błędu nie została zmieniona
for x, y in test_gen:
  print("min:", x[0].min(), "max:", x[0].max())
  plt.title(labels[np.argmax(y[0])])
  plt.imshow(x[0])
  plt.show()
  break
# przetestowaliśmy generator

# stwórzmy generator, którego użyjemy podczas treningu
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)

# stwórzmy generator, którego użyjemy podczas walidacji
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


# Keras poza standardową metodą fit (która przyjmuje array numpy)
# posiada metodę fit_generator, która przyjmuje generator.
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)


# Funkcja generująca macierz niepewności
def get_confusion_matrix(data_path, N):
  # scikit learn posiada taką funkcję, ale
  # mamy problem, bo tej funkcji mamy przekazać
  # cele i przewidywania (żeby porównała) a tego
  # nie mamy, bo wszystko co mamy to generatory.
  # Ale możemy użyć genretarów, żeby zbudować te
  # potrzebne tablice. I to robi ta funkcja
  print("Generating confusion matrix", N)
  predictions = []  # przewidywania
  targets = []  # cele
  i = 0

  # Iterujemy i generujemy
  for x, y in gen.flow_from_directory(
          data_path,
          target_size=IMAGE_SIZE,
          shuffle=False,
          batch_size=batch_size * 2
  ):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)

# Wyświetly jakieś wyniki
# Błąd
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Dokładność
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

from util import plot_confusion_matrix
plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')