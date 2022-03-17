import torch
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        imgs_remapped = sample['images_remapped']
        imgs_remapped_new = []
        for img_remapped in imgs_remapped:
            img_remapped = np.array(img_remapped).astype(np.float32)
            mask = np.array(mask).astype(np.float32)
            img_remapped /= 255.0
            img_remapped -= self.mean
            img_remapped /= self.std
            imgs_remapped_new.append(img_remapped)

        return {'image': img,
                'images_remapped': imgs_remapped_new,
                'label': mask,
                'u_map': sample['u_map'],
                'v_map': sample['v_map'],
                'mask' : sample['mask']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        remapmask = sample['mask']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        remapmask = torch.from_numpy(remapmask)

        imgs_remapped = sample['images_remapped']
        imgs_remapped_new=[]
        for img_remapped in imgs_remapped:
            img_remapped = np.array(img_remapped).astype(np.float32).transpose((2, 0, 1))
            img_remapped = torch.from_numpy(img_remapped).float()
            imgs_remapped_new.append(img_remapped)

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map = torch.from_numpy(u_map.astype(np.float32)).float()
        v_map = torch.from_numpy(v_map.astype(np.float32)).float()

        return {'image': img,
                'images_remapped': imgs_remapped_new,
                'label': mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask' : remapmask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        imgs_remapped = sample['images_remapped']
        u_map = sample['u_map']
        v_map = sample['v_map']
        remapmask = sample['mask']
        imgs_remapped_new = []
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            u_map = u_map[:,::-1]
            remapmask = remapmask[:,::-1]
            for img_remapped in imgs_remapped:
                img_remapped = img_remapped.transpose(Image.FLIP_LEFT_RIGHT)
                imgs_remapped_new.append(img_remapped)
        else:
            imgs_remapped_new = imgs_remapped

        return {'image': img,
                'images_remapped': imgs_remapped_new,
                'label': mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask' : remapmask}


# class RandomRotate(object):
#     def __init__(self, degree):
#         self.degree = degree

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         rotate_degree = random.uniform(-1*self.degree, self.degree)
#         img = img.rotate(rotate_degree, Image.BILINEAR)
#         mask = mask.rotate(rotate_degree, Image.NEAREST)

#         imgs_remapped = sample['images_remapped']
#         imgs_remapped_new = []
#         for img_remapped in imgs_remapped:
#             img_remapped = img_remapped.rotate(rotate_degree, Image.BILINEAR)
#             imgs_remapped_new.append(img_remapped)

#         return {'image': img,
#                 'images_remapped': imgs_remapped_new,
#                 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        imgs_remapped = sample['images_remapped']
        imgs_remapped_new = []
        if random.random() < 0.5:
            radius = random.random()
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            for img_remapped in imgs_remapped:
                img_remapped = img_remapped.filter(ImageFilter.GaussianBlur(radius=radius))
                imgs_remapped_new.append(img_remapped)
        else:
            imgs_remapped_new = imgs_remapped

        return {'image': img,
                'images_remapped': imgs_remapped_new,
                'label': mask,
                'u_map': sample['u_map'],
                'v_map': sample['v_map'],
                'mask' : sample['mask']}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        remapmask = sample['mask']
        remapmask = Image.fromarray(remapmask)

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            remapmask = ImageOps.expand(remapmask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        remapmask = remapmask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        imgs_remapped = sample['images_remapped']
        imgs_remapped_new = []
        for img_remapped in imgs_remapped:
            img_remapped = img_remapped.resize((ow, oh), Image.BILINEAR)
            if short_size < self.crop_size:
                padh = self.crop_size - oh if oh < self.crop_size else 0
                padw = self.crop_size - ow if ow < self.crop_size else 0
                img_remapped = ImageOps.expand(img_remapped, border=(0, 0, padw, padh), fill=0)
            # random crop crop_size
            img_remapped = img_remapped.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            imgs_remapped_new.append(img_remapped)

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map = cv2.resize(u_map,(ow,oh))
        v_map = cv2.resize(v_map,(ow,oh))
        if short_size < self.crop_size:
            u_map_ = np.zeros((oh+padh,ow+padw))
            u_map_[:oh,:ow] = u_map
            u_map = u_map_
            v_map_ = np.zeros((oh+padh,ow+padw))
            v_map_[:oh,:ow] = v_map
            v_map = v_map_
        u_map = u_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        v_map = v_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        
        remapmask = np.array(remapmask)
        assert remapmask.dtype == np.bool
        return {'image': img,
                'images_remapped': imgs_remapped_new,
                'label': mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask' : remapmask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        remapmask = sample['mask']
        remapmask = Image.fromarray(remapmask)

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        remapmask = remapmask.resize((ow, oh), Image.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        remapmask = remapmask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        imgs_remapped = sample['images_remapped']
        imgs_remapped_new = []
        for img_remapped in imgs_remapped:
            img_remapped = img_remapped.resize((ow, oh), Image.BILINEAR)
            img_remapped = img_remapped.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            imgs_remapped_new.append(img_remapped)

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map = cv2.resize(u_map,(ow,oh))
        v_map = cv2.resize(v_map,(ow,oh))
        u_map = u_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        v_map = v_map[y1:y1+self.crop_size, x1:x1+self.crop_size]

        remapmask = np.array(remapmask)
        return {'image': img,
                'images_remapped': imgs_remapped_new,
                'label': mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask' : remapmask}

# class FixedResize(object):
#     def __init__(self, size):
#         self.size = (size, size)  # size: (h, w)

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']

#         assert img.size == mask.size

#         img = img.resize(self.size, Image.BILINEAR)
#         mask = mask.resize(self.size, Image.NEAREST)

#         imgs_remapped = sample['images_remapped']
#         imgs_remapped_new = []
#         for img_remapped in imgs_remapped:
#             img_remapped = img_remapped.resize(self.size, Image.BILINEAR)
#             imgs_remapped_new.appned(img_remapped)

#         return {'image': img,
#                 'image_remapped': imgs_remapped_new,
#                 'label': mask}