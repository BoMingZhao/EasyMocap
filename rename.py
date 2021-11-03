from genericpath import exists
import os


'''dataset_dir = 'D:/boming/human_pose/Nerual_body_dataset/CoreView_386'
for i in range(1, 24):
    image_path = os.path.join(dataset_dir, 'Camera_B' + str(i))
    j = 0
    for file in os.listdir(image_path):
        old_name = os.path.join(image_path, file)
        new_dir = os.path.join(dataset_dir, str('%.2d' % i))
        os.makedirs(new_dir, exist_ok=True)
        new_name = os.path.join(new_dir, str('%.6d' % j) + '.jpg')
        os.rename(old_name, new_name)
        j += 1'''

dataset_dir = 'D:/boming/human_pose/Nerual_body_dataset/387/mask'
for i in range(1, 24):
    old_name = os.path.join(dataset_dir, 'Camera_B' + str(i))
    new_name = os.path.join(dataset_dir, str('%.2d' % i))
    os.rename(old_name, new_name)