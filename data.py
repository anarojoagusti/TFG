from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import nibabel as nib
#import tables
#import cv2
#from libtiff import TIFF
#path_ana = 'C:/Users/anaro/OneDrive/Escritorio/liver/nifti'

class myAugmentation(object):
	
	"""
	A class used to augmentate image
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""
	def __init__(self, train_path='train/image', label_path='train/label', merge_path='train/merge', aug_merge_path='train/aug_merge', aug_train_path='train/aug_train', aug_label_path='train/aug_label', img_type="nii"):
		
		"""
		Using glob to get all .img_type form path
		"""

		self.train_imgs = glob.glob("/*."+img_type)
		self.label_imgs = glob.glob("/*."+img_type)
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.slices = len(self.train_imgs)
		self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

	def Augmentation(self):

		"""
		Start augmentation.....
		"""
		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path
		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print ("trains can't match labels")
			return 0
		for i in range(len(trains)):
			img_t = load_img(path_train+"/"+str(i)+"."+imgtype)
			img_l = load_img(path_label+"/"+str(i)+"."+imgtype)
			x_t = img_to_array(img_t)
			x_l = img_to_array(img_l)
			x_t[:,:,2] = x_l[:,:,0]
			img_tmp = array_to_img(x_t)
			img_tmp.save(path_merge+"/"+str(i)+"."+imgtype)
			img = x_t
			img = img.reshape((1,) + img.shape)
			savedir = path_aug_merge + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.doAugmentate(img, savedir, str(i))


	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='.nii', imgnum=30):

		"""
		augmentate one image
		"""
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
		    i += 1
		    if i > imgnum:
		        break

	def splitMerge(self):

		"""
		split merged image apart
		"""
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path
		for i in range(self.slices):
			path = path_merge + "/" + str(i)
			train_imgs = glob.glob(path+"/*."+self.img_type)
			savedir = path_train + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			savedir = path_label + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
	#		for imgname in train_imgs:
	#			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
	#			img = cv2.imread(imgname)
	#			img_train = img[:,:,2]#cv2 read image rgb->bgr
	#			img_label = img[:,:,0]
	#			cv2.imwrite(path_train+"/"+str(i)+"/"+midname+"_train"+"."+self.img_type,img_train)
#				cv2.imwrite(path_label+"/"+str(i)+"/"+midname+"_label"+"."+self.img_type,img_label)

	def splitTransform(self):

		"""
		split perspective transform images
		"""
		#path_merge = "transform"
		#path_train = "transform/data/"
		#path_label = "transform/label/"
		path_merge = "train/merge"
		path_train = "train/image"
		path_label = "train/label"
		train_imgs = glob.glob(path_merge+"/*."+self.img_type)
		for imgname in train_imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
			img = cv2.imread(imgname)
			img_train = img[:,:,2]#cv2 read image rgb->bgr
			img_label = img[:,:,0]
			cv2.imwrite(path_train+midname+"."+self.img_type,img_train)
			cv2.imwrite(path_label+midname+"."+self.img_type,img_label)



class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path = "C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train/image_a", label_path = "C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train/label_a", test_path = "C:/Users/anaro/OneDrive/Escritorio/liver/niftis/test", npy_path = "C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train", img_type = "nii"):

		"""
		
		"""
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path
        
        
	def create_train_data(self):
		 
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		img_ini = np.zeros(shape=(0,192,192))
		imgs = os.listdir("C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train/image_a")
		print(len(imgs))
		
		for i in range(len(imgs)):
			
			ima = nib.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train/image_a/'+ imgs[i])
			img = ima.get_data()
			img = np.array(img)
			img = np.transpose(img,(2,0,1))
			img = np.transpose(img,(0,2,1))
			maxi = np.max(img)
			mini = np.min(img)
			denom = maxi - mini
			img = (img-mini)/denom
			img_ini=np.concatenate((img_ini,img), axis=0)
			imgdatas=img_ini[...,np.newaxis]
		print('loading images done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		print('Saving images to .npy files done')
   
		label_ini = np.zeros(shape=(0,192,192))
		labls = os.listdir('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train/label_a')
		print(len(labls))
		for i in range(len(labls)):
			label = nib.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train/label_a/'+ labls[i])
			label = label.get_data()
			label = np.array(label)
			label = np.transpose(label, (2,0,1))
			label = np.transpose(label, (0,2,1))
			label[label<0.5] = 0
			label[label>0.5] = 1 
			label_ini=np.concatenate((label_ini,label), axis=0)  		
			imglabels=label_ini[..., np.newaxis]
		print('loading masks done')
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving masks to .npy files done.')

	def create_test_data(self):
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		test_ini = np.zeros(shape=(0,192,192))
		tests = os.listdir("C:/Users/anaro/OneDrive/Escritorio/liver/niftis/test6/")
		print(len(tests))      
		for i in range(len(tests)):
			test = nib.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/test6/'+ tests[i])
			Aff_rot2 = test.affine 
			test = test.get_data()
			test = np.array(test)
			test = np.transpose(test, (2,0,1))
			test = np.transpose(test, (0,2,1))
			maxit = np.max(test)
			minit = np.min(test)
			denomt = maxit - minit
			test = (test-minit)/denomt
			test_ini=np.concatenate((test_ini,test), axis=0)
			imgtests=test_ini[..., np.newaxis]
		print('loading done')
		np.save(self.npy_path + '/imgs_test6.npy', imgtests)
		print('Saving to imgs_test.npy files done.')
        
		labelt_ini = np.zeros(shape=(0,192,192))
		lablst = os.listdir('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/label test6')
		print(len(lablst))
		for i in range(len(lablst)):
			labelt = nib.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/label test6/'+ lablst[i])
			labelt = labelt.get_data()
			labelt = np.array(labelt)
			labelt = np.transpose(labelt, (2,0,1))
			labelt = np.transpose(labelt, (0,2,1))
			labelt[labelt<0.5] = 0
			labelt[labelt>0.5] = 1 
			labelt_ini=np.concatenate((labelt_ini,labelt), axis=0)  		
		imgs_gth = np.transpose(labelt_ini, (1,2,0))	
		print('loading masks done')
		np.save(self.npy_path + '/imgs_mask_gth6.npy', imgs_gth)
		print('Saving masks to .npy files done.')
        

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train'+'/imgs_train.npy')
		imgs_mask_train = np.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train'+ '/imgs_mask_train.npy')
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
#		xmax=np.amax(imgs_train, axis=0)
		#maxim = imgs_train.sort()
		#maxi = maxim(len(maxim))
		#mean = imgs_train.mean(axis = 0)
		#imgs_train -= mean	
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train'+'/imgs_test6.npy')
		imgs_test = imgs_test.astype('float32')
		#mean = imgs_test.mean(axis = 0)
		#imgs_test -= mean	
		return imgs_test

if __name__ == "__main__":

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	mydata = dataProcess(192,192)
	mydata.create_train_data()
	mydata.create_test_data()
	imgs_train,imgs_mask_train= mydata.load_train_data()
	imgs_test= mydata.load_test_data()
	print (imgs_train.shape,imgs_mask_train.shape)

