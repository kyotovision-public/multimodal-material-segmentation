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
        # img /= 255.0
        img -= self.mean
        img /= self.std

        nir = sample['nir']
        nir = np.array(nir).astype(np.float32)
        # nir /= 255

        return {'image': img,
                'label': mask,
                'aolp' : sample['aolp'], 
                'dolp' : sample['dolp'], 
                'nir'  : nir, 
                'nir_mask': sample['nir_mask'],
                'u_map': sample['u_map'],
                'v_map': sample['v_map'],
                'mask':sample['mask']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        aolp = sample['aolp']
        dolp = sample['dolp']
        nir  = sample['nir']
        nir_mask  = sample['nir_mask']
        SS=sample['mask']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        aolp = np.array(aolp).astype(np.float32).transpose((2, 0, 1))
        dolp = np.array(dolp).astype(np.float32)
        SS = np.array(SS).astype(np.float32)
        nir = np.array(nir).astype(np.float32)
        nir_mask = np.array(nir_mask).astype(np.float32)
        
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        aolp = torch.from_numpy(aolp).float()
        dolp = torch.from_numpy(dolp).float()
        SS = torch.from_numpy(SS).float()
        nir = torch.from_numpy(nir).float()
        nir_mask = torch.from_numpy(nir_mask).float()

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map = torch.from_numpy(u_map.astype(np.float32)).float()
        v_map = torch.from_numpy(v_map.astype(np.float32)).float()

        return {'image': img,
                'label': mask,
                'aolp' : aolp,
                'dolp' : dolp,
                'nir'  : nir,
                'nir_mask'  : nir_mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask':SS}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        aolp = sample['aolp']
        dolp = sample['dolp']
        nir  = sample['nir']
        nir_mask  = sample['nir_mask']
        u_map = sample['u_map']
        v_map = sample['v_map']
        SS=sample['mask']
        if random.random() < 0.5:
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # nir = nir.transpose(Image.FLIP_LEFT_RIGHT)

            img = img[:,::-1]
            mask = mask[:,::-1]
            nir = nir[:,::-1]
            nir_mask = nir_mask[:,::-1]
            aolp  = aolp[:,::-1]
            dolp  = dolp[:,::-1]
            SS  = SS[:,::-1]
            u_map = u_map[:,::-1]

        return {'image': img,
                'label': mask,
                'aolp' : aolp,
                'dolp' : dolp,
                'nir'  : nir,
                'nir_mask'  : nir_mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask':SS}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        nir  = sample['nir']
        if random.random() < 0.5:
            radius = random.random()
            # img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            # nir = nir.filter(ImageFilter.GaussianBlur(radius=radius))
            img = cv2.GaussianBlur(img, (0,0), radius)
            nir = cv2.GaussianBlur(nir, (0,0), radius)

        return {'image': img,
                'label': mask,
                'aolp' : sample['aolp'], 
                'dolp' : sample['dolp'], 
                'nir'  : nir, 
                'nir_mask': sample['nir_mask'],
                'u_map': sample['u_map'],
                'v_map': sample['v_map'],
                'mask':sample['mask']}

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        aolp = sample['aolp']
        dolp = sample['dolp']
        nir = sample['nir']
        nir_mask = sample['nir_mask']
        SS=sample['mask']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # w, h = img.size
        h, w = img.shape[:2]
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            
        # random crop crop_size
        # w, h = img.size
        h, w = img.shape[:2]

        # x1 = random.randint(0, w - self.crop_size)
        # y1 = random.randint(0, h - self.crop_size)
        x1 = random.randint(0, max(0, ow - self.crop_size))
        y1 = random.randint(0, max(0, oh - self.crop_size))

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map    = cv2.resize(u_map,(ow,oh))
        v_map    = cv2.resize(v_map,(ow,oh))
        aolp     = cv2.resize(aolp ,(ow,oh))
        dolp     = cv2.resize(dolp ,(ow,oh))
        SS     = cv2.resize(SS ,(ow,oh))
        img      = cv2.resize(img  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
        mask     = cv2.resize(mask ,(ow,oh), interpolation=cv2.INTER_NEAREST)
        nir      = cv2.resize(nir  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
        nir_mask = cv2.resize(nir_mask  ,(ow,oh), interpolation=cv2.INTER_NEAREST)
        if short_size < self.crop_size:
            u_map_ = np.zeros((oh+padh,ow+padw))
            u_map_[:oh,:ow] = u_map
            u_map = u_map_
            v_map_ = np.zeros((oh+padh,ow+padw))
            v_map_[:oh,:ow] = v_map
            v_map = v_map_
            aolp_ = np.zeros((oh+padh,ow+padw,2))
            aolp_[:oh,:ow] = aolp
            aolp = aolp_
            dolp_ = np.zeros((oh+padh,ow+padw))
            dolp_[:oh,:ow] = dolp
            dolp = dolp_

            img_ = np.zeros((oh+padh,ow+padw,3))
            img_[:oh,:ow] = img
            img = img_
            SS_ = np.zeros((oh+padh,ow+padw))
            SS_[:oh,:ow] = SS
            SS = SS_
            mask_ = np.full((oh+padh,ow+padw),self.fill)
            mask_[:oh,:ow] = mask
            mask = mask_
            nir_ = np.zeros((oh+padh,ow+padw))
            nir_[:oh,:ow] = nir
            nir = nir_
            nir_mask_ = np.zeros((oh+padh,ow+padw))
            nir_mask_[:oh,:ow] = nir_mask
            nir_mask = nir_mask_

        u_map = u_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        v_map = v_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        aolp  =  aolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
        dolp  =  dolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
        img   =   img[y1:y1+self.crop_size, x1:x1+self.crop_size]
        mask  =  mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        nir   =   nir[y1:y1+self.crop_size, x1:x1+self.crop_size]
        SS   =   SS[y1:y1+self.crop_size, x1:x1+self.crop_size]
        nir_mask = nir_mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        return {'image': img,
                'label': mask,
                'aolp' : aolp,
                'dolp' : dolp,
                'nir'  : nir,
                'nir_mask'  : nir_mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask':SS}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        aolp = sample['aolp']
        dolp = sample['dolp']
        nir = sample['nir']
        nir_mask = sample['nir_mask']
        SS = sample['mask']

        # w, h = img.size
        h, w = img.shape[:2]

        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # nir = nir.resize((ow, oh), Image.BILINEAR)

        # center crop
        # w, h = img.size
        # h, w = img.shape[:2]
        x1 = int(round((ow - self.crop_size) / 2.))
        y1 = int(round((oh - self.crop_size) / 2.))
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # nir = nir.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        u_map = sample['u_map']
        v_map = sample['v_map']
        u_map = cv2.resize(u_map,(ow,oh))
        v_map = cv2.resize(v_map,(ow,oh))
        aolp  = cv2.resize(aolp ,(ow,oh))
        dolp  = cv2.resize(dolp ,(ow,oh))
        SS  = cv2.resize(SS ,(ow,oh))
        img   = cv2.resize(img  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask ,(ow,oh), interpolation=cv2.INTER_NEAREST)
        nir   = cv2.resize(nir  ,(ow,oh), interpolation=cv2.INTER_LINEAR)
        nir_mask = cv2.resize(nir_mask,(ow,oh), interpolation=cv2.INTER_NEAREST)
        u_map = u_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        v_map = v_map[y1:y1+self.crop_size, x1:x1+self.crop_size]
        aolp  =  aolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
        dolp  =  dolp[y1:y1+self.crop_size, x1:x1+self.crop_size]
        img   =   img[y1:y1+self.crop_size, x1:x1+self.crop_size]
        mask  =  mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        SS  =  SS[y1:y1+self.crop_size, x1:x1+self.crop_size]
        nir   =   nir[y1:y1+self.crop_size, x1:x1+self.crop_size]
        nir_mask = nir_mask[y1:y1+self.crop_size, x1:x1+self.crop_size]
        return {'image': img,
                'label': mask,
                'aolp' : aolp,
                'dolp' : dolp,
                'nir'  : nir,
                'nir_mask'  : nir_mask,
                'u_map': u_map,
                'v_map': v_map,
                'mask':SS}