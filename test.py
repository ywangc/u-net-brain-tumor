# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:59:53 2017

@author: ywang
"""
import model
import tensorflow as tf
import numpy as np
import tensorlayer as tl
import os
import nibabel as nib
batch_size = 1
nw = 240
nh = 240
nz = 4
def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
        image_path=path)
    
path = "E:/unet/checkpoint/"
params = tl.files.load_npz(path, name = 'u_net.npz')
x = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
y = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment')
network  = model.u_net(x, is_train = False, reuse = True)
with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.files.assign_params(sess, params, network)
    
test_data_path = "E:/unet/testdata"
test_path_list = tl.files.load_folder_list(path=test_data_path)
test_name_list = [os.path.basename(p) for p in test_path_list]
index_test = list(range(0, len(test_name_list)))
data_types = ['flair', 't1', 't1ce', 't2']
data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}
for i in data_types:
    data_temp_list = []
    for j in test_name_list:
        img_path = os.path.join(test_data_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data()
        data_temp_list.append(img)
        data_temp_list = np.asarray(data_temp_list)
    m = np.mean(data_temp_list)
    s = np.std(data_temp_list)
    data_types_mean_std_dict[i]['mean'] = m
    data_types_mean_std_dict[i]['std'] = s
del data_temp_list
print(data_types_mean_std_dict)
X_test_input = []
X_test_label = []
for i in test_name_list:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(test_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float32)
        all_3d_data.append(img)
        seg_path = os.path.join(test_data_path, i, i + '_seg.nii.gz')
        seg_img = nib.load(seg_path).get_data()
        seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_test_input.append(combined_array)
        seg_2d = seg_img[:, :, j]
        seg_2d.astype(int)
        X_test_label.append(seg_2d)
        
X_test_input = np.asarray(X_test_input, dtype=np.float32)
X_test_label = np.asarray(X_test_label, dtype=np.float32)

test = X_test_input[np.newaxis,96,:,:,:]
test_seg = X_test_label[np.newaxis,96,:,:,np.newaxis]
out_seg = network.outputs
dice_loss = 1 - tl.cost.dice_coe(out_seg, test_seg, axis=[0,1,2,3])
_dice, pred = sess.run([dice_loss,network.outputs],feed_dict={x: test, y:test_seg})
test = test.reshape([240,240,4])
test_seg = test_seg.reshape([240,240,1])
pred= pred.reshape([240,240,1])
vis_imgs2(test, test_seg, pred, "E:/unet/sample/test-96.png")

