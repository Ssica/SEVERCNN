from __future__ import division
import os
import sys
import SimpleITK as itk
import numpy as np
import skimage
from skimage import filters
from scipy import ndimage
import scipy
import skimage.io as io

'''
returns a tuple of 5 elements for each modality,
a list of paths for each patient.
'''
def _files(dir):
    OT = []
    Flair = []
    T1 = []
    T1c = []
    T2 = []
    _all = []
    for root, dirs, files in os.walk(dir):
        for name in files:
                p = os.path.join(root, name)
                if ".mha" in p:
                    _all.append(p)
                    if ".OT." in p:
                        OT.append(p)
                    elif ".MR_Flair." in p:
                        Flair.append(p)
                    elif ".MR_T1." in p:
                        T1.append(p)
                    elif ".MR_T1c." in p:
                        T1c.append(p)
                    elif ".MR_T2." in p:
                        T2.append(p)
    return (OT, Flair, T1, T1c, T2, _all)

OT,Flair,T1,T1c, T2, _all  = _files("./BRATS_EDEMA/BRATS_TRAIN/")
#OT, Flair, T1, T1c, T2, _all = _files("./Testing/HGG_LGG/")
print "fn: " + T1[0]
img = itk.ReadImage(T1[10])
new_arr = itk.GetArrayFromImage(img)
a = new_arr[:,1,:]
b = new_arr[:,2,:]
c = np.dstack((a,b))
print new_arr.shape
print np.shape(a)
print np.shape(b)
print np.shape(c)

ot = itk.ReadImage(OT[0])
flair = itk.ReadImage(Flair[0])
t1 = itk.ReadImage(T1[0])
t2 = itk.ReadImage(T2[0])
t1c = itk.ReadImage(T1c[0])

out_img = io.imread(T1c[0], plugin='simpleitk')
io.imsave('./t1c_test.nii', out_img, plugin='simpleitk')

in_img = io.imread('./t1c_test.nii', plugin='simpleitk')
print in_img.shape
print np.sum(in_img)

input_img = t1c
mask_img = itk.OtsuThreshold( input_img, 0, 1, 200)
input_img = itk.Cast(input_img, itk.sitkFloat32)
corrector = itk.N4BiasFieldCorrectionImageFilter()
numberFilltingLevels = 4
output = corrector.Execute(input_img, mask_img)

io.imsave('./bias_corrected.mha', plugin='simpleitk')
"""
print "OT:"
print itk.GetArrayFromImage(ot).shape
print "Flair:"
print itk.GetArrayFromImage(flair).shape
print "T1:"
print itk.GetArrayFromImage(t1).shape
print "T2:"
print itk.GetArrayFromImage(t2).shape

tmp = scipy.ndimage.interpolation.zoom(itk.GetArrayFromImage(t1), [(240/155), 1.0,1.0])
tmp2 = scipy.ndimage.interpolation.zoom(itk.GetArrayFromImage(t1c), [(240/155), 1.0, 1.0])
tmp3 = scipy.ndimage.interpolation.zoom(itk.GetArrayFromImage(t2), [240/155,1.0,1.0])
tmp4 = scipy.ndimage.interpolation.zoom(itk.GetArrayFromImage(flair), [240/155, 1.0, 1.0])

tmp = np.moveaxis(tmp, 0, 1)
tmp2 = np.moveaxis(tmp2, 0, 1)
tmp3 = np.moveaxis(tmp3, 0, 1)
tmp4 = np.moveaxis(tmp4, 0, 1)

la = np.append(tmp[:,:,:,np.newaxis], tmp2[:,:,:,np.newaxis], axis=3)
la = np.append(la, tmp3[:,:,:,np.newaxis], axis=3)
la = np.append(la, tmp4[:,:,:,np.newaxis], axis=3)
print la.shape

print np.concatenate((la, la), axis=0).shape
"""
"""
print "3D: "
print new_arr.shape
print np.shape(new_arr[:,6,:])
print "2D: "
print new_arr[0, :,:].shape
print "sum of array:"
print np.sum(new_arr)
"""

def read_files(dir):
    (_,_,T1,_,_,paths) = _files(dir)
    ret_list = []
    for x in T1:
        img = itk.ReadImage(x)
        _data = itk.GetArrayFromImage(img)
	inputsize = img.GetSize() 
        ret_list.append((img, _data, inputsize))
    return ret_list

'''
la = read_files("./BRATS_EDEMA/BRATS_TRAIN/")
print la[0]
'''
