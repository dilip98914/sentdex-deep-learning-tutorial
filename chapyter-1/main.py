import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def main():
	# loading datasets and testsets
	mnist=keras.datasets.mnist
	(x_train,y_train),(x_test,y_test)=mnist.load_data()

	#normalizing datasets
	x_train=keras.utils.normalize(x_train,axis=1)
	x_test=keras.utils.normalize(x_test,axis=1)


	# model creation
	model=keras.models.Sequential()

	# adding layers
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128,activation='relu'))
	model.add(keras.layers.Dense(128,activation='relu'))
	model.add(keras.layers.Dense(10,activation='softmax'))

	#compiling the neural network
	model.compile(optimizer='adam',
					loss='sparse_categorical_crossentropy',
					metrics=['accuracy'])

	#train or fit
	model.fit(x_train,y_train,epochs=1)

	#evaluation over test data
	val_loss,val_acc=model.evaluate(x_test,y_test)

	print('loss',val_loss)
	print('accuracy',val_acc)

	#saving the neural_network/model
	model_json=model.to_json()
	with open('model.json','w') as f:
		f.write(model_json)
	model.save_weights('model.h5')
	# model.save('num_classifier.model')


	model.load_weights('model.h5')

	

	predictions=model.predict(x_test)
	# print(predictions)
	print('prediction: ',np.argmax(predictions[0]))
	plt.imshow(x_test[0],cmap=plt.cm.binary)
	plt.show()

if __name__=='__main__':
	main()
