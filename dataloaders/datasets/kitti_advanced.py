from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
# from dataloaders import custom_transforms as tr
from dataloaders import custom_transforms_adv as tr

class KITTIAdvSegmentation(Dataset):
    # NUM_CLASSES = 21
    NUM_CLASSES = 20

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('kitti_advanced'),
                 split='train',
                 ):
        """
        :param base_dir: path to KITTI dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'train', 'image_2')
        # print("@@@@@@@@@@@@@@@@@@ USE sequences_remapped @@@@@@@@@@@@@@@@@@@")
        # self._image_remapped_dir = os.path.join(self._base_dir, 'sequences_remapped')
        # print("@@@@@@@@@@@@@@@@@@ USE sequences_remapped2 @@@@@@@@@@@@@@@@@@@")
        # self._image_remapped_dir = os.path.join(self._base_dir, 'sequences_remapped2')
        print("@@@@@@@@@@@@@@@@@@ USE sequences_remapped4 @@@@@@@@@@@@@@@@@@@")
        self._image_remapped_dir = os.path.join(self._base_dir, 'sequences_remapped4')
        self._cat_dir = os.path.join(self._base_dir, 'train', 'semantic')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, self.args.list_folder)

        self.im_ids = []
        self.images = []
        self.images_remapped = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                scenename = line.rsplit('_',1)[0]
                frame_id = int(line.rsplit('_',1)[1])
                # if frame_id == 0 or frame_id == 245:
                #     continue
                _image = os.path.join(self._image_dir, line + ".png")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                _images_remapped = []
                for i in range(args.propagation):
                    i += 1
                    _image_remapped = os.path.join(self._image_remapped_dir, scenename, f"{frame_id-i:010}+{i}.png")
                    if not os.path.isfile(_image_remapped):
                        break
                    # assert os.path.isfile(_image_remapped)
                    _images_remapped.append(_image_remapped)
                    _image_remapped = os.path.join(self._image_remapped_dir, scenename, f"{frame_id+i:010}-{i}.png")
                    # assert os.path.isfile(_image_remapped)
                    if not os.path.isfile(_image_remapped):
                        break
                    _images_remapped.append(_image_remapped)
                else:
                    self.im_ids.append(line)
                    self.images.append(_image)
                    self.categories.append(_cat)
                    self.images_remapped.append(_images_remapped)
                    assert (len(_images_remapped) == 2 * args.propagation)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

        self.img_h = 320
        self.img_w = 1216
        max_dim = max(self.img_h, self.img_w)
        u_vec = (np.arange(self.img_w)-self.img_w/2)/max_dim*2
        v_vec = (np.arange(self.img_h)-self.img_h/2)/max_dim*2
        self.u_map, self.v_map = np.meshgrid(u_vec, v_vec)
        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _imgs_remapped, _target, _mask = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'images_remapped': _imgs_remapped, 'label': _target, 'u_map': self.u_map, 'v_map': self.v_map, 'mask':_mask}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        _imgs_remapped = []
        w, h = _img.size
        _mask = np.zeros((h,w),dtype=np.bool)
        for filename in self.images_remapped[index]:
            _img_remapped = Image.open(filename).convert('RGB')
            _mask += (np.sum(np.array(_img_remapped),axis=2)==0)
            _imgs_remapped.append(_img_remapped)
        return _img, _imgs_remapped, _target, _mask

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


