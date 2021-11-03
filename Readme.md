1. download https://zjueducn-my.sharepoint.com/personal/pengsida_zju_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fneural%5Fbody%2Fdataset%2Fzju%5Fmocap
You can check if there is no intri file in 313 and 315
2. Move all camera_images to dataset_dir/images/ and rename them as '%.2d' % i, i=1~23
3. Move intri.yml and extri.yml to dataset_dir/
4. Move mask_cihp to dataset_dir and rename as 'mask', notice that the data in mask_cihp should rename as '%.2d' % i, i=1~23
5. Move keypoints2d to dataset_dir and rename as 'opnepose', notice that the data in keypoints2d should rename as '%.2d' % i, i=1~23
6. Run python scripts/preprocess/extract_video.py dataset_dir --handface
7. Run python apps/demo/mv1p.py dataset_dir --outdataset_dir/smpl --vis_det --vis_repro --undis --sub_vis 01 07 13 19 --vis_smpl
8. Run RenderPeople_generate.py