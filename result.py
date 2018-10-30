import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Conv3D,MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import numpy as np

filepath1 = '/home/chunwei/Documents/class/CPE520/HW8/model.h5'
labeltext = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

epochs = 100
batch_size = 50
learningRate = 0.01

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0 #To make the data between 0~1
y_test_label = y_test
y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)

x_train, x_validation = x_train[0:40000],x_train[40000:50000]
y_train, y_validation = y_train[0:40000],y_train[40000:50000]

model  = tf.keras.models.load_model(
	    filepath1,
	    custom_objects=None,
	    compile=True
	)

score = model.evaluate(x_test, y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
predict_classes = model.predict_classes(x_test)


result = np.zeros((10,10), dtype=int)
result_index = np.zeros(10, dtype=int)
full_label = 0
index = 0

while full_label < 10 and index <= predict_classes.size:
	label = predict_classes[index]
	if result_index[label] <= 9:
		print(label)
		print(result_index[label])
		result[label][result_index[label]] = index
		result_index[label] = result_index[label] + 1
		if result_index[label] > 9:
			full_label = full_label + 1
	index = index + 1	

fig, axarr = plt.subplots(10,10)
for x in xrange(0,10):
	for y in xrange(0,10):
		axarr[x,y].imshow(x_test[result[x][y]].reshape(32,32,3),cmap=plt.cm.binary) 
		# plt.subplot(10,10,x*10+y+1)
		axarr[x,y].axis('off')
		# plt.imshow(x_test[result[x][y]].reshape(32,32,3),cmap=plt.cm.binary)   

for ax in axarr.flat:
    ax.set_ylabel(ylabel='y-label')
    ax.label_outer()



# plt.figure(figsize=(12,9))
# for j in xrange(0,3):
# 	for i in xrange(0,4):
# 		plt.subplot(4,3,j*4+i+1)
# 		plt.title('predict:{}/real:{}'.format(predict_classes[j*4+i] ,y_test_label[j*4+i]))
# 		plt.axis('off')
# 		plt.imshow(x_test[j*4+i].reshape(32,32,3),cmap=plt.cm.binary)   
fig.subplots_adjust(left=0.15)
plt.show()