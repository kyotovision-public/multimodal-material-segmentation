from __future__ import print_function, division
import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms_multimodal_old as tr

class HandmadeDatasetSegmentation(Dataset):
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('handmade_dataset_stereo'),
                 split='train',
                 ):
        """
        :param base_dir: path to KITTI dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'RGB')
        self._cat_dir = os.path.join(self._base_dir, 'semantic')
        self._aolp_sin_dir = os.path.join(self._base_dir, 'AoLP_sin')
        self._aolp_cos_dir = os.path.join(self._base_dir, 'AoLP_cos')
        self._dolp_dir = os.path.join(self._base_dir, 'DoLP')
        self._nir_dir = os.path.join(self._base_dir, 'NIR_warped')
        self._left_offset = 192

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, self.args.list_folder)

        self.im_ids = []
        self.images = []
        self.aolp_sins  = []
        self.aolp_coss  = []
        self.dolps  = []
        self.nirs   = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".png")
                _cat   = os.path.join(self._cat_dir  , line + ".png")
                _aolp_sin  = os.path.join(self._aolp_sin_dir , line + ".npy")
                _aolp_cos  = os.path.join(self._aolp_cos_dir , line + ".npy")
                _dolp  = os.path.join(self._dolp_dir , line + ".npy")
                _nir   = os.path.join(self._nir_dir  , line + ".png")

                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.aolp_sins.append(_aolp_sin)
                self.aolp_coss.append(_aolp_cos)
                self.dolps.append(_dolp)
                self.nirs.append(_nir)
                
        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

        self.img_h = 1024
        self.img_w = 1224
        max_dim = max(self.img_h, self.img_w)
        u_vec = (np.arange(self.img_w)-self.img_w/2)/max_dim*2
        v_vec = (np.arange(self.img_h)-self.img_h/2)/max_dim*2
        self.u_map, self.v_map = np.meshgrid(u_vec, v_vec)
        self.u_map = self.u_map[:,:self._left_offset]
        self.v_map = self.v_map[:,:self._left_offset]
        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target, _aolp, _dolp, _nir = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'aolp': _aolp, 'dolp': _dolp, 'nir': _nir, 'u_map': self.u_map, 'v_map': self.v_map}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = cv2.imread(self.images[index],-1)
        _img = _img.astype(np.float32)/65535 if _img.dtype==np.uint16 else _img.astype(np.float32)/255
        _target = cv2.imread(self.categories[index],-1)
        _aolp_sin = np.load(self.aolp_sins[index])
        _aolp_cos = np.load(self.aolp_coss[index])
        _aolp = np.stack([_aolp_sin, _aolp_cos], axis=2) # H x W x 2
        _dolp = np.load(self.dolps[index])
        _nir  = cv2.imread(self.nirs[index],-1)
        _nir = _nir.astype(np.float32)/65535 if _nir.dtype==np.uint16 else _nir.astype(np.float32)/255
        return _img[:,:self._left_offset], _target[:,:self._left_offset], \
               _aolp[:,:self._left_offset], _dolp[:,:self._left_offset], \
               _nir[:,:self._left_offset]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'KITTI_material_dataset(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
    # plt.savefig('./out.png')


