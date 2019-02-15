import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D


pickle_in=open('x.pickle','rb')
x=pickle.load(pickle_in)


pickle_in=open('y.pickle','rb')
y=pickle.load(pickle_in)





model=Sequential()
model.add(Conv2D(256,(3,3),input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#this converts our 3d feature map into 1d feautre map

model.add(Flatten())

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
model.fit(x,y,batch_size=32,epochs=3,validation_split=0.3)


# base_dir=os.path.join(os.getcwd(),'pets')
# categories=['Dog','Cat']
# img_size=50

# for category in categories:
# 	path=os.path.join(base_dir,category)#create path to dogs and cats 
# 	for img in os.listdir(path):
# 		#convert to array
# 		img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
# 		new_array=cv2.resize(img_array,(img_size,img_size))
# 		plt.imshow(new_array,cmap='gray')
# 		plt.show()
# 		break
# 	break


# print(img_array)
# print(img_array.shape)

# training_data=[]

# def create_training_date():
# 	for category in categories:
# 		path=os.path.join(base_dir,category)
# 		class_num=categories.index(category)

# 		for img in tqdm(os.listdir(path)):
# 			try:
# 				img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
# 				new_array=cv2.resize(img_array,(img_size,img_size))
# 				# new_array=cv2.resize(img_array,(img_size,img_size))
# 				training_data.append([new_array,class_num])
# 			except Exception as e:
# 				print('this image is broke')



# create_training_date()



# # print('length'+len(training_data))				
# random.shuffle(training_data)
# x=[]
# y=[]
	
# for features,label in training_data:
# 	x.append(features)
# 	y.append(label)
# x=np.array(x).reshape(-1,img_size,img_size,1)

# # pickle_out=open('x.pickle','wb')
# # pickle.dump(x,pickle_out)
# # pickle_out.close()

# # pickle_out=open('y.pickle','wb')
# # pickle.dump(y,pickle_out)
# # pickle_out.close()




