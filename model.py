from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D,Input
from keras.models import Model,save_model,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from keras.optimizers import Adam,SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50,Xception,InceptionResNetV2

import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from glob import glob


class TBDetection():
	"""docstring for TBDetection"""
	def __init__(self):
		self.img_rows,self.img_cols = 256,256
		self.batch_size = 16
		self.seed = 42
		self.num_classes = 2
		self.input_shape = (self.img_rows,self.img_cols,3)

		self.fold_name = 'fold2'
		self.train_img_path = 'data/folds/'+self.fold_name+'/train/'
		self.validation_img_path = 'data/folds/'+self.fold_name+'/validation/'

		self.train_generator = self.get_train_generator(self.train_img_path)
		self.validation_generator = self.get_validation_generator(self.validation_img_path)
		self.train_samples = len(self.train_generator.filenames)
		self.validation_samples = len(self.validation_generator.filenames)

		self.name = 'ResNet50'
		self.save_path = 'models/'+self.name+'_best.h5'

		self.model = self.get_model()

	def get_model(self):

		base_model = ResNet50(include_top=False,input_shape=self.input_shape,weights='imagenet')
		# base_model = Xception(include_top=False,input_shape=self.input_shape,weights='imagenet')
		# base_model = InceptionResNetV2(include_top=False,input_shape=self.input_shape,weights='imagenet')

		x = base_model.output
		x = Flatten()(x)
		x = Dense(512,activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(512,activation='relu')(x)
		x = Dropout(0.5)(x)
		predictions = Dense(self.num_classes,activation='softmax')(x)
		
		model = Model(base_model.inputs,predictions)
		return model


	def build_model(self,lr=1e-4):
		
		opt = Adam(lr=lr)
		self.model.compile(
			optimizer = opt,
			loss = 'binary_crossentropy',
			metrics = ['accuracy']
			)


	def get_train_generator(self,path):
		img_gen = ImageDataGenerator(
			zoom_range = 0.2,
			width_shift_range = 0.2,
			height_shift_range = 0.2,
			horizontal_flip = False,
			rotation_range = 10.0,
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb'
			)

		return img_gen

	def get_validation_generator(self,path):
		img_gen = ImageDataGenerator(
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed =self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb',
			shuffle = False
			)

		return img_gen

	def get_callbacks(self):
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		
		return [early_stopping,checkpointer,tensorboard]
		# return [checkpointer,tensorboard]

	def train(self,lr=1e-4,num_epochs=2):

		self.build_model(lr)

		train_generator = self.train_generator
		validation_generator = self.validation_generator

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			)

		return hist

	def continue_training(self,lr=1e-4,num_epochs=2):

		self.model = load_model(self.save_path)
		self.build_model(lr)

		train_generator = self.train_generator
		validation_generator = self.validation_generator

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs  = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			)

		return hist


	def get_predictions(self,save=True):
		
		model = load_model(self.save_path)
		validation_generator = self.validation_generator

		y_pred = model.predict_generator(
			generator = validation_generator,
			steps = self.validation_samples//self.batch_size + 1,
			verbose = 1,
			)

		y_true = validation_generator.classes

		if save:
			np.save('tmp/'+self.name+'_pred.npy',y_pred)
			np.save('tmp/'+self.name+'_true.npy',y_true)

		return y_pred,y_true

	def get_metrics(self):
		y_pred = np.load('tmp/'+self.name+'_pred.npy')
		y_actual = np.load('tmp/'+self.name+'_true.npy')

		print y_pred.shape
		print y_actual.shape
		
		y_pred = np.argmax(y_pred,axis=1)

		cm = confusion_matrix(y_actual,y_pred)
		report = classification_report(y_actual,y_pred)
		accuracy = accuracy_score(y_actual,y_pred)

		print cm
		print report
		print 'Accuracy :',accuracy


if __name__ == '__main__':
	t1 = TBDetection()
	t1.train(lr=1e-4,num_epochs=20)
	# t1.continue_training(lr=1e-4,num_epochs=1)
	t1.get_predictions(save=True)
	t1.get_metrics()