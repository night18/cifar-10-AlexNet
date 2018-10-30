import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Conv3D,MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import numpy as np

#The path of the stored model
filepath1 = '/home/chunwei/Documents/class/CPE520/HW8/model.h5'
filepath2 = '/home/chunwei/Documents/class/CPE520/HW8/model2.h5'
filepath3 = '/home/chunwei/Documents/class/CPE520/HW8/model3.h5'
filepath4 = '/home/chunwei/Documents/class/CPE520/HW8/model4.h5'


epochs = 100
batch_size = 50
learningRate = 0.01


def plot_loss(history, history2, history3, history4):
	plt.subplot(2,2,1)
	plt.plot(history.history['val_acc'])
	plt.plot(history2.history['val_acc'])
	plt.plot(history3.history['val_acc'])
	plt.plot(history4.history['val_acc'])

	plt.xticks(np.arange(0,epochs, (epochs/10)))
	plt.title('Val accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['Q1', 'Q2', 'Q3', "Q4"], loc=0)

	plt.subplot(2,2,2)
	plt.plot(history.history['val_loss'])
	plt.plot(history2.history['val_loss'])
	plt.plot(history3.history['val_loss'])
	plt.plot(history4.history['val_loss'])

	plt.xticks(np.arange(0,epochs, (epochs/10) ))
	plt.title('Val loss')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(['Q1', 'Q2', 'Q3', "Q4"], loc=0)

	plt.subplot(2,2,3)
	plt.plot(history.history['acc'])
	plt.plot(history2.history['acc'])
	plt.plot(history3.history['acc'])
	plt.plot(history4.history['acc'])

	plt.xticks(np.arange(0,epochs, (epochs/10) ))
	plt.title('Train accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['Q1', 'Q2', 'Q3', "Q4"], loc=0)

	plt.subplot(2,2,4)
	plt.plot(history.history['loss'])
	plt.plot(history2.history['loss'])
	plt.plot(history3.history['loss'])
	plt.plot(history4.history['loss'])

	plt.xticks(np.arange(0,epochs, (epochs/10) ))
	plt.title('Train loss')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(['Q1', 'Q2', 'Q3', "Q4"], loc=0)

	plt.show()


#RGB
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0 #To make the data between 0~1
y_test_label = y_test
y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)

x_train, x_validation = x_train[0:40000],x_train[40000:50000]
y_train, y_validation = y_train[0:40000],y_train[40000:50000]

print(x_train.shape, 'train samples')
print(y_train.shape, 'train labels')
print(x_validation.shape, 'validation samples')
print(y_validation.shape, 'validation labels')
print(x_test.shape, 'test samples')
print(y_test.shape, 'train labels')
print(x_train[0].shape)

# try:
# 	model  = tf.keras.models.load_model(
# 	    filepath1,
# 	    custom_objects=None,
# 	    compile=True
# 	)
# except Exception as e:
model = Sequential()

#Layer 1 
model.add( Conv2D(48, kernel_size=(3,3),strides=(1,1), activation='relu', padding='same', input_shape=x_train.shape[1:] ) )
model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 2
model.add( Conv2D(96, kernel_size=(3,3), activation='relu', padding='same') )
model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 3
model.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )


#Layer 4
model.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )
model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 5
model.add( Conv2D(256, kernel_size=(3,3), activation='relu', padding='same') )
model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

model.add(Flatten())

#Layer 6
model.add(Dense(512, activation='tanh'))

#Layer 7 
model.add(Dense(256, activation='tanh'))

#Prediction
model.add(Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=SGD(lr=learningRate),
              metrics=['accuracy'])

csv_logger1 = CSVLogger('log1.csv', append=True, separator=';')
hist = model.fit(x_train, y_train,
          epochs=epochs,
          batch_size= batch_size,
          validation_data=(x_validation, y_validation),
          callbacks=[csv_logger1])

tf.keras.models.save_model(
	model,
	filepath1,
	overwrite=True,
	include_optimizer=True
	)


model2 = Sequential()

#Layer 1 
model2.add( Conv2D(48, kernel_size=(3,3),strides=(1,1), activation='relu', padding='same', input_shape=x_train.shape[1:] ) )
model2.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 2
model2.add( Conv2D(96, kernel_size=(3,3), activation='relu', padding='same') )
model2.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 3
model2.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )


#Layer 4
model2.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )
model2.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

model2.add(Flatten())

#Layer 6
model2.add(Dense(512, activation='tanh'))

#Layer 7 
model2.add(Dense(256, activation='tanh'))

#Prediction
model2.add(Dense(10, activation='softmax'))

model2.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=SGD(lr=learningRate),
              metrics=['accuracy'])

csv_logger2 = CSVLogger('log2.csv', append=True, separator=';')
hist2 = model2.fit(x_train, y_train,
          epochs=epochs,
          batch_size= batch_size,
          validation_data=(x_validation, y_validation),
          callbacks=[csv_logger2])

tf.keras.models.save_model(
	model2,
	filepath2,
	overwrite=True,
	include_optimizer=True
	)



model3 = Sequential()

#Layer 1 
model3.add( Conv2D(48, kernel_size=(3,3),strides=(1,1), activation='relu', padding='same', input_shape=x_train.shape[1:] ) )
model3.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 2
model3.add( Conv2D(96, kernel_size=(3,3), activation='relu', padding='same') )
model3.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 3
model3.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )

#Layer 5
model3.add( Conv2D(256, kernel_size=(3,3), activation='relu', padding='same') )
model3.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

model3.add(Flatten())

#Layer 6
model3.add(Dense(512, activation='tanh'))

#Layer 7 
model3.add(Dense(256, activation='tanh'))

#Prediction
model3.add(Dense(10, activation='softmax'))

model3.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=SGD(lr=learningRate),
              metrics=['accuracy'])

csv_logger3 = CSVLogger('log3.csv', append=True, separator=';')
hist3 = model3.fit(x_train, y_train,
          epochs=epochs,
          batch_size= batch_size,
          validation_data=(x_validation, y_validation),
          callbacks=[csv_logger3])


tf.keras.models.save_model(
	model3,
	filepath3,
	overwrite=True,
	include_optimizer=True
	)

model4 = Sequential()

#Layer 1 
model4.add( Conv2D(48, kernel_size=(3,3),strides=(1,1), activation='relu', padding='same', input_shape=x_train.shape[1:] ) )
model4.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 2
model4.add( Conv2D(96, kernel_size=(3,3), activation='relu', padding='same') )
model4.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 3
model4.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )


#Layer 4
model4.add( Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )
model4.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

#Layer 5
model4.add( Conv2D(256, kernel_size=(3,3), activation='relu', padding='same') )
model4.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )

model4.add(Flatten())


#Layer 7 
model4.add(Dense(256, activation='tanh'))

#Prediction
model4.add(Dense(10, activation='softmax'))

model4.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=SGD(lr=learningRate),
              metrics=['accuracy'])

csv_logger4 = CSVLogger('log4.csv', append=True, separator=';')
hist4 = model4.fit(x_train, y_train,
          epochs=epochs,
          batch_size= batch_size,
          validation_data=(x_validation, y_validation),
          callbacks=[csv_logger4])

tf.keras.models.save_model(
	model4,
	filepath4,
	overwrite=True,
	include_optimizer=True
	)

plot_loss(hist, hist2, hist3, hist4)



# finally:
# 	score = model.evaluate(x_test, y_test)

# 	print('Test loss:', score[0])
# 	print('Test accuracy:', score[1])
# 	predict_classes = model.predict_classes(x_test)

	# plt.figure(figsize=(12,9))
	# for j in xrange(0,3):
	# 	for i in xrange(0,4):
	# 		plt.subplot(4,3,j*4+i+1)
	# 		plt.title('predict:{}/real:{}'.format(predict_classes[j*4+i] ,y_test_label[j*4+i]))
	# 		plt.axis('off')
	# 		plt.imshow(x_test[j*4+i].reshape(32,32,3),cmap=plt.cm.binary)   

	# plt.show()




