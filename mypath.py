class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return './datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'kitti':
            return './datasets/KITTI_material_dataset.old/'  # folder that contains KITTI_material_dataset.
        elif dataset == 'kitti_advanced':
            return './datasets/KITTI_material/'  # folder that contains KITTI_material_dataset.
        elif dataset == 'kitti_advanced_manta':
            return '/home/wakaki/manta_local/KITTI_material/'  # folder that contains KITTI_material_dataset.
        elif dataset == 'handmade_dataset':
            return './datasets/handmade_dataset_for_train/'  # folder that contains KITTI_material_dataset.
        elif dataset == 'handmade_dataset_stereo':
            return './datasets/handmade_dataset_for_train_stereo/'  # folder that contains KITTI_material_dataset.
        elif dataset == 'multimodal_dataset':
            return './datasets/multimodal_dataset/'  # folder that contains KITTI_material_dataset.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
