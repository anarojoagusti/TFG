from unet import *
from data import *
import nibabel as nib
import numpy as np

mydata = dataProcess(192,192)

imgs_test = mydata.load_test_data()

myUnet = myUnet()

model = myUnet.get_unet()

model.load_weights('unet.hdf15')

imgs_mask_test = model.predict(imgs_test, verbose=1)

np.save('imgs_mask_test156.npy', imgs_mask_test)

masks = np.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/imgs_mask_test156.npy')

for i in range(masks.shape[3]):
    mask = np.array(masks)
    mask = mask[:,:,:,i]
    mask[mask<0.49] = 0
    mask[mask>0.49] = 1
    maskara = np.transpose(mask, (1,2,0))
imagen = nib.Nifti1Image(maskara, affine=Aff_rot2)
nib.save(imagen, 'C:/Users/anaro/OneDrive/Escritorio/liver/niftis/results/imgs_mask_test156.nii')
np.save('imgs_mask_pred156.npy', maskara)


gth = np.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/train/imgs_mask_gth6.npy')
pred = np.load('C:/Users/anaro/OneDrive/Escritorio/liver/niftis/imgs_mask_pred156.npy')
gth_s = gth.shape
pred_s = pred.shape

# Calc DICE coeff
dice_coeff = (2*(np.sum(gth[pred>0.99])))/(np.sum(gth[gth>0.99])+np.sum(pred[pred>0.99]))