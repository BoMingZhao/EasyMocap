import json
import numpy as np
import os
import cv2
import torch
import pickle

from torch._C import device
'''for each_sample:
        'id': index. int
        'subject_name': name of the subject. str
        'action_name': name of the action. str
        'smplx_shapes': smplx shape. np
        'smpl_shapes': smpl shape. np
        'cameras': list of dict for camera parameters.
        for each_cam:
            'Tc2w': Camera pose. Camera to world transform. np.array 4x4
            'K': Camera intrinsic. np.array 3x3
            'dist': Camera distortion. np array 1x5
            'name': Camera name. str
            'frames': A list of frames. list
            for each_frame:
                'image_path': Image file path from dataset_dir
                'smpl_pose': 25x3 
                'smplx_pose': 29x3
                'smpl_RT': RT for people space to camera space'''
device='cuda'
class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        else:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            cv2.FileStorage.write(self.fs, key, value)
        elif dt == 'list':
            if self.major_version == 4: # 4.4
                self.fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for elem in value:
                    self.fs.write('', elem)
                self.fs.endWriteStruct()
            else: # 3.4
                self.fs.write(key, '[')
                for elem in value:
                    self.fs.write('none', elem)
                self.fs.write('none', ']')

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['T'] = Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    return cams

dataset_list = [386, 387]
'''for i in range(9):
    x = 363 + i
    dataset_list.append(x)
for i in range(11):
    x = 377 + i
    dataset_list.append(x)'''
dataset_dir = 'D:/boming/human_pose/Nerual_body_dataset/rp_kumar_rigged_real'
bin_path = 'D:/boming/human_pose/Nerual_body_dataset/rp_kumar_rigged_real/rp_kumar_rigged_real.bin'
sample = []

for dataset in dataset_list:
    sample_input = {}
    path = os.path.join(dataset_dir, str(dataset))
    sample_input['id'] = dataset
    sample_input['subject_name'] = 'zju_mocap_smpl_smplx'
    sample_input['action_name'] = ''
    cameras = []
    intri_name = os.path.join(path, 'intri.yml')
    extri_name = os.path.join(path, 'extri.yml')
    cam = read_camera(intri_name, extri_name)
    for i, value in cam.items():
        i = int(i)
        cam_dict = {}
        RT = np.concatenate((value['RT'], np.array([0, 0, 0, 1]).reshape([1, 4])))
        reverse = np.array([[1, 0 ,0 ,0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0 , 0, 1]])
        w2c = np.dot(np.linalg.inv(reverse), RT)
        c2w = np.linalg.inv(w2c)
        cam_dict['Tc2w'] = c2w
        cam_dict['K'] = value['K']
        cam_dict['dist'] = value['dist']
        cam_dict['name'] = str(i)
        frame_list = []
        l = len(os.listdir(os.path.join(path, 'images', '01')))
        for frame in range(l):
            frame_dist = {}
            #smplx_file_name = os.path.join(path, 'smplx', ('%.6d' % i) + '.json')
            smpl_file_name = os.path.join(path, 'smpl', 'smpl', ('%.6d' % frame) + '.json')
            #smplx_json = json.load(open(smplx_file_name))
            smpl_json = json.load(open(smpl_file_name))

            #smplx_poses = np.array(smplx_json[0]['poses'][0]).reshape([29, 3])
            #smplx_pose_list = []
            '''for k in range(29):
                smplx_pose_list.append(cv2.Rodrigues(smplx_poses[k])[0].reshape([1, 3, 3]))
            smplx_pose = np.concatenate(smplx_pose_list)'''

            smpl_poses = np.array(smpl_json[0]['poses'][0]).reshape([24, 3])
            smpl_pose_list = []
            for k in range(24):
                smpl_pose_list.append(cv2.Rodrigues(smpl_poses[k])[0].reshape([1, 3, 3]))
            smpl_pose = np.concatenate(smpl_pose_list)

            '''smplx_R = cv2.Rodrigues(np.array(smplx_json[0]['Rh'][0]))[0].reshape([3, 3])
            smplx_RT = np.concatenate([smplx_R, np.array(smplx_json[0]['Th'][0]).reshape([3, 1])], axis=1)
            smplx_RT = np.concatenate([smplx_RT, np.array([0, 0, 0, 1]).reshape(1, 4)])
            smplx_shapes = np.array(smplx_json[0]['shapes'][0])'''

            smpl_R = cv2.Rodrigues(np.array(smpl_json[0]['Rh'][0]))[0].reshape([3, 3])
            smpl_RT = np.concatenate([smpl_R, np.array(smpl_json[0]['Th'][0]).reshape([3, 1])], axis=1)
            smpl_RT = np.concatenate([smpl_RT, np.array([0, 0, 0, 1]).reshape(1, 4)])
            smpl_shapes = np.array(smpl_json[0]['shapes'][0])

            if frame == 0: #first time save the shape
                #sample_input['smplx_shapes'] = smplx_shapes
                sample_input['smpl_shapes'] = smpl_shapes
            frame_dist['image_path'] = os.path.join('rp_kumar_rigged_real', str(dataset), 'images', ('%.2d' % i), ('%.6d' % frame) + '.jpg').replace('\\', '/')
            '''frame_dist['smplx_pose'] = smplx_pose
            frame_dist['smplx_RT'] = smplx_RT'''
            frame_dist['smpl_pose'] = smpl_pose
            frame_dist['smpl_RT'] = smpl_RT
            mask_path = os.path.join(dataset_dir, str(dataset), 'mask', ('%.2d' % i), ('%.6d' % frame) + '.png')
            depth_path = os.path.join('rp_kumar_rigged_real', str(dataset), 'depth', ('%.2d' % i), ('%.6d' % frame) + '.npz').replace('\\', '/')
            os.makedirs(os.path.join(dataset_dir, str(dataset), 'depth', ('%.2d' % i)), exist_ok=True)
            mask = torch.from_numpy(cv2.imread(mask_path)).to(device)
            mask = torch.sum(mask, dim=-1)
            zero = torch.zeros_like(mask).float()
            one = torch.ones_like(mask).float()
            mask = torch.where(mask == 0, zero, one).cpu().numpy()
            np.savez_compressed(os.path.join('D:/boming/human_pose/Nerual_body_dataset', depth_path), mat=mask)
            frame_dist['depth_path'] = depth_path
            frame_list.append(frame_dist)
        cam_dict['frames'] = frame_list
        cameras.append(cam_dict)
        #camera_num = i + 1 # we need 0 ~ k, the data is 1 ~ k + 1
    sample_input['cameras'] = cameras
    sample.append(sample_input)

with open(bin_path, 'wb') as f:
    pickle.dump(sample, f)
    f.close()