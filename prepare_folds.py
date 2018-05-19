import os
import shutil
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split,KFold


def create_folder(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)

def create_folder_skeleton(base_path,n_splits=4):
	for i in range(n_splits):
		fold_path = base_path+'fold'+str(i+1)+'/'
		skeleton = ['train/normal/','train/active/','validation/normal/','validation/active/']
		for item in skeleton:
			create_folder(fold_path+item)

def get_folds(n_splits=4):

	df = pd.read_csv('data/csv/chinatb.csv')
	X = df['Image']
	y = df['TYPE']

	folds = KFold(n_splits=n_splits,random_state=42)
	for train_index,test_index in folds.split(X,y):
		train_X = X[train_index]
		train_y = y[train_index]
		test_X = X[test_index]
		test_y = y[test_index]
		
		yield train_X,train_y,test_X,test_y

def copy_images(in_paths,out_paths):
	for src,dst in zip(in_paths,out_paths):
		shutil.copy(src,dst)

def copy_fold_images(base_path,n_splits):

	folds = get_folds(n_splits)

	for i in range(n_splits):

		train_X,train_y,test_X,test_y = folds.next()
		
		fold_path = base_path+'fold'+str(i+1)+'/'

		img_in_paths = [''.join(['data/all/cropped/',name,'.png']) for name in train_X]
		img_out_paths = [''.join([fold_path,'train/',category.lower(),'/',name,'.png']) for category,name in zip(train_y,train_X)]
		copy_images(img_in_paths,img_out_paths)

		img_in_paths = [''.join(['data/all/cropped/',name,'.png']) for name in train_X]
		img_out_paths = [''.join([fold_path,'validation/',category.lower(),'/',name,'.png']) for category,name in zip(test_y,test_X)]
		copy_images(img_in_paths,img_out_paths)
		
def get_stats(base_path):
	for root, dirs,filenames in os.walk(base_path):
		if len(filenames)>0:
			print root,":",len(filenames)

def put_gitignores(base_path):
	for root,dirs,filenames in os.walk(base_path):
		if not dirs:
			content = "*.png\n*.jpg\n*.jpeg"
			path = root+'/.gitgnore'
			file = open(path,'w')
			file.write(content)
			print path
			# os.remove(path)


if __name__ == '__main__':
	# create_folder_skeleton(base_path='data/folds/',n_splits=4)
	# copy_fold_images(base_path='data/folds/',n_splits=4)
	# put_gitignores(base_path='data/')
	get_stats(base_path='data/folds/')

