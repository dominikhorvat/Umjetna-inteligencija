import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import os
from keras.layers import Dropout
from PIL import Image

#postavljamo trenutni direktorij na glavni
os.chdir(os.path.dirname(os.path.abspath(__file__)))

mnist=tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()

pozeljno = 784 
x_train = x_train.reshape(60000, pozeljno) 
x_test = x_test.reshape(10000, pozeljno) 

x_train = x_train.astype('float32')
x_test = x_test.astype('float32') 

x_train /= 255
x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, 10) 
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(512, input_shape=(784,), activation = 'relu'))
model.add(Dropout(0.4))

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(tf.keras.layers.Dense(10, activation = 'softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
training = model.fit(x_train, y_train, batch_size=256, epochs=32)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

model.save('mreza_python.keras')

print("Uspjesno spremljen model!")

model=tf.keras.models.load_model('mreza_python.keras')


#--------------Prvi direktorij sa slikama od 0 do 4-------------------------

#izrada slika za testiranje
broj_slike=1

while os.path.isfile(f"znamenke/0_do_4/znamenka{broj_slike}.png"):
    original_slika = Image.open(f"znamenke/0_do_4/znamenka{broj_slike}.png")
    # Promeni dimenzije na 28x28
    resized_slika = original_slika.resize((28, 28))
 
    # Sačuvaj promenjenu sliku
    resized_slika.save(f"testiranje/0_do_4/podesena{broj_slike}.png")
    broj_slike +=1

print("Uspjesno prilagodjene slike!")

#stvaranje polja za podatke, prepoznavanje, ispis
za_testiranje = []
broj_slika=1

while os.path.isfile(f"testiranje/0_do_4/podesena{broj_slika}.png"):
    img = cv2.imread(f"testiranje/0_do_4/podesena{broj_slika}.png", cv2.IMREAD_GRAYSCALE)
    za_testiranje.append(np.array(img))
#na kraju povećamo varijablu broj_slika za 1
    broj_slika +=1

temp_array = np.invert(np.array(za_testiranje))
array_flat = temp_array.reshape(len(temp_array), 28*28)
array_flat = array_flat / 255
za_pretpostaviti = model.predict(array_flat)
pretpostavka = [np.argmax(i) for i in za_pretpostaviti]


# Iteriraj kroz slike
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
# Iteriraj kroz slike
i=0
for j in range(1, broj_slika):
    # Kreiraj naziv slike
   
    putanja_slike = os.path.join("testiranje/0_do_4", f"podesena{j}.png")

    # Učitaj sliku
    img = Image.open(putanja_slike)

    # Invertiraj boje
    inverted_img = Image.eval(img, lambda x: 255 - x)

    # Odredi indekse redaka i stupaca za prikaz
    redak = (j - 1) // 5
    stupac = (j - 1) % 5
    # Prikazi sliku na odgovarajućem mjestu u subplotu
    axs[redak, stupac].imshow(inverted_img, cmap='gray')
    axs[redak, stupac].set_title("Pretpostavka:" +str(pretpostavka[i]))
    axs[redak, stupac].axis('off')  # Isključi oznake osi
    i=i+1


plt.show()


#izbrisi slike koje su bile za testiranje

za_obrisati=1

#uklanjamo sve podesene slike velicina 28x28 piksela
while os.path.isfile(f"testiranje/0_do_4/podesena{za_obrisati}.png"):
    datoteka = (f"testiranje/0_do_4/podesena{za_obrisati}.png")
    os.remove(datoteka)
    za_obrisati +=1

print("Uspjesno obrisane slike!")

#--------------Drugi direktorij sa slikama od 5 do 9-------------------------

#izrada slika za testiranje
broj_slike=1

while os.path.isfile(f"znamenke/5_do_9/znamenka{broj_slike}.png"):
    original_slika = Image.open(f"znamenke/5_do_9/znamenka{broj_slike}.png")
    # Promeni dimenzije na 28x28
    resized_slika = original_slika.resize((28, 28))
 
    # Sačuvaj promenjenu sliku
    resized_slika.save(f"testiranje/5_do_9/podesena{broj_slike}.png")
    broj_slike +=1

print("Uspjesno prilagodjene slike!")

#stvaranje polja za podatke, prepoznavanje, ispis
za_testiranje = []
broj_slika=1

while os.path.isfile(f"testiranje/5_do_9/podesena{broj_slika}.png"):
    img = cv2.imread(f"testiranje/5_do_9/podesena{broj_slika}.png", cv2.IMREAD_GRAYSCALE)
    za_testiranje.append(np.array(img))
#na kraju povećamo varijablu broj_slika za 1
    broj_slika +=1

temp_array = np.invert(np.array(za_testiranje))
array_flat = temp_array.reshape(len(temp_array), 28*28)
array_flat = array_flat / 255
za_pretpostaviti = model.predict(array_flat)
pretpostavka = [np.argmax(i) for i in za_pretpostaviti]


# Iteriraj kroz slike
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
# Iteriraj kroz slike
i=0
for j in range(1, broj_slika):
    # Kreiraj naziv slike
   
    putanja_slike = os.path.join("testiranje/5_do_9", f"podesena{j}.png")

    # Učitaj sliku
    img = Image.open(putanja_slike)

    # Invertiraj boje
    inverted_img = Image.eval(img, lambda x: 255 - x)

    # Odredi indekse redaka i stupaca za prikaz
    redak = (j - 1) // 5
    stupac = (j - 1) % 5
    # Prikazi sliku na odgovarajućem mjestu u subplotu
    axs[redak, stupac].imshow(inverted_img, cmap='gray')
    axs[redak, stupac].set_title("Pretpostavka:" +str(pretpostavka[i]))
    axs[redak, stupac].axis('off')  # Isključi oznake osi
    i=i+1


plt.show()

#izbrisi slike koje su bile za testiranje

za_obrisati=1

#uklanjamo sve podesene slike velicina 28x28 piksela
while os.path.isfile(f"testiranje/5_do_9/podesena{za_obrisati}.png"):
    datoteka = (f"testiranje/5_do_9/podesena{za_obrisati}.png")
    os.remove(datoteka)
    za_obrisati +=1

print("Uspjesno obrisane slike!")

#AUTOR: Dominik Horvat
#Kolegij: Umjetna inteligencija
#Prirodoslovno matematicki fakultet u Zagrebu