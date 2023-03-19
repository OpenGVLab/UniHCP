import numpy as np
import cv2
import os
import random
from PIL import Image

import torch
import torchvision.transforms as T
from helper.align import kestrel_get_similar_matrix

cv2.ocl.setUseOpenCL(False)

class FaceAugmentation(object):
    def __init__(self, crop_size, final_size, crop_center_y_offset, scale_aug, trans_aug):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip
    def __call__(self, img):
        ## transform
        scale_diff_h = (np.random.rand()*2-1)*self.scale_aug
        scale_diff_w = (np.random.rand()*2-1)*self.scale_aug
        crop_aug_h = self.crop_size*(1+scale_diff_h)
        crop_aug_w = self.crop_size*(1+scale_diff_w)

        trans_diff_h = (np.random.rand()*2-1)*self.trans_aug
        trans_diff_w = (np.random.rand()*2-1)*self.trans_aug

        w, h = img.size
        ct_x = w/2*(1+trans_diff_w)
        ct_y = (h/2+self.crop_center_y_offset)*(1+trans_diff_h)

        if ct_x < crop_aug_w/2:
            crop_aug_w = ct_x*2 - 0.5
        if ct_y < crop_aug_h/2:
            crop_aug_h = ct_y*2 - 0.5
        if ct_x + crop_aug_w/2 >= w:
            crop_aug_w = (w-ct_x)*2 - 0.5
        if ct_y + crop_aug_h/2 >= h:
            crop_aug_h = (h-ct_y)*2 - 0.5

        rect = (ct_x-crop_aug_w/2, ct_y-crop_aug_h/2, ct_x+crop_aug_w/2, ct_y+crop_aug_h/2)
        img = img.resize((self.final_size, self.final_size), box=rect)

        ## to BGR
        img = np.array(img)
        img = img[:,:,[2,1,0]]

        return img

class FaceAugmentationCV2(object):
    def __init__(self, crop_size, final_size, crop_center_x_offset, \
                crop_center_y_offset, scale_aug, trans_aug, flip=-1, \
                mask_aug=0, half=0):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.crop_center_x_offset = crop_center_x_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip
        self.mask_aug = mask_aug
        self.half = half

    def __call__(self, img):
        ## transform
        scale_diff_h = (np.random.rand()*2-1)*self.scale_aug
        scale_diff_w = (np.random.rand()*2-1)*self.scale_aug
        crop_aug_h = self.crop_size*(1+scale_diff_h)
        crop_aug_w = self.crop_size*(1+scale_diff_w)

        trans_diff_h = (np.random.rand()*2-1)*self.trans_aug
        trans_diff_w = (np.random.rand()*2-1)*self.trans_aug

        h, w, _ = img.shape
        ct_x = (w/2+self.crop_center_x_offset)*(1+trans_diff_w)
        ct_y = (h/2+self.crop_center_y_offset)*(1+trans_diff_h)

        if ct_x < crop_aug_w/2:
            crop_aug_w = ct_x*2 - 0.5
        if ct_y < crop_aug_h/2:
            crop_aug_h = ct_y*2 - 0.5
        if ct_x + crop_aug_w/2 >= w:
            crop_aug_w = (w-ct_x)*2 - 0.5
        if ct_y + crop_aug_h/2 >= h:
            crop_aug_h = (h-ct_y)*2 - 0.5

        #rect = (ct_x-crop_aug_w/2, ct_y-crop_aug_h/2, ct_x+crop_aug_w/2, ct_y+crop_aug_h/2)
        #img = img.resize((self.final_size, self.final_size), box=rect)
        t = int(np.ceil(ct_y-crop_aug_h/2))
        #d = int(np.ceil(ct_y+crop_aug_h/2))
        l = int(np.ceil(ct_x-crop_aug_w/2))
        #r = int(np.ceil(ct_x+crop_aug_w/2))
        img = img[t:int(t+crop_aug_h),l:int(l+crop_aug_w),:]

        img = cv2.resize(img, (self.final_size, self.final_size))

        if self.half == 1:

            img[:self.final_size // 2, :] = 0

        elif self.half == -1:

            img[self.final_size // 2 :, :] = 0

        #self.mask_aug = 1
        if self.mask_aug > 0:

            seed = np.random.rand()
            aug = self.final_size // 2 * min(np.random.rand() + 0.1, 1)
            if seed < self.mask_aug / 2:
                img[:int(self.final_size//2-aug), :] = 0
            elif seed < self.mask_aug:
                img[int(self.final_size//2+aug):, :] = 0

        #print(self.flip)
        if np.random.rand() <= self.flip:
            #print('do flip')
            img = cv2.flip(img, 1)

        ## to BGR
        #img = np.array(img)
        #img = img[:,:,[2,1,0]]

        return img

class FaceAugmentationCV2Mask(object):
    def __init__(self, crop_size, final_size, crop_center_y_offset, scale_aug, trans_aug, mask_spec,
                total_ratio,sunglass,sunglass_ratio,mask,mask_ratio,hat,hat_ratio,flip=-1,mask_type='random'):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip
        #####add mask_spec by wl
        self.mask_spec = mask_spec
        self.total_ratio = total_ratio
        self.sunglass = sunglass
        self.sunglass_ratio = sunglass_ratio
        self.mask = mask
        self.mask_ratio = mask_ratio
        self.hat = hat
        self.hat_ratio = hat_ratio

    def __call__(self, img):
        ## transform
        scale_diff_h = (np.random.rand()*2-1)*self.scale_aug
        scale_diff_w = (np.random.rand()*2-1)*self.scale_aug
        crop_aug_h = self.crop_size*(1+scale_diff_h)
        crop_aug_w = self.crop_size*(1+scale_diff_w)

        trans_diff_h = (np.random.rand()*2-1)*self.trans_aug
        trans_diff_w = (np.random.rand()*2-1)*self.trans_aug

        h, w, _ = img.shape
        ct_x = w/2*(1+trans_diff_w)
        ct_y = (h/2+self.crop_center_y_offset)*(1+trans_diff_h)

        if ct_x < crop_aug_w/2:
            crop_aug_w = ct_x*2 - 0.5
        if ct_y < crop_aug_h/2:
            crop_aug_h = ct_y*2 - 0.5
        if ct_x + crop_aug_w/2 >= w:
            crop_aug_w = (w-ct_x)*2 - 0.5
        if ct_y + crop_aug_h/2 >= h:
            crop_aug_h = (h-ct_y)*2 - 0.5

        #rect = (ct_x-crop_aug_w/2, ct_y-crop_aug_h/2, ct_x+crop_aug_w/2, ct_y+crop_aug_h/2)
        #img = img.resize((self.final_size, self.final_size), box=rect)
        t = int(np.ceil(ct_y-crop_aug_h/2))
        #d = int(np.ceil(ct_y+crop_aug_h/2))
        l = int(np.ceil(ct_x-crop_aug_w/2))
        #r = int(np.ceil(ct_x+crop_aug_w/2))
        img = img[t:int(t+crop_aug_h),l:int(l+crop_aug_w),:]
        img = cv2.resize(img, (self.final_size, self.final_size))

        #print(self.flip)
        if np.random.rand() <= self.flip:
            #print('do flip')
            img = cv2.flip(img, 1)

        ## to BGR
        #img = np.array(img)
        #img = img[:,:,[2,1,0]]
        if self.mask_spec == True:
            LEx = 70.7
            LEy = 113.0
            REx = 108.23
            REy = 113.0
            Mx = 89.43
            My = 153.51
            LEx_mod = (LEx - (ct_x - crop_aug_w/2))*(self.final_size/crop_aug_w)
            LEy_mod = (LEy - (ct_y - crop_aug_h/2))*(self.final_size/crop_aug_h)
            REx_mod = (REx - (ct_x - crop_aug_w/2))*(self.final_size/crop_aug_w)
            REy_mod = (REy - (ct_y - crop_aug_h/2))*(self.final_size/crop_aug_h)
            Mx_mod = (Mx - (ct_x - crop_aug_w/2))*(self.final_size/crop_aug_w)
            My_mod = (My - (ct_y - crop_aug_h/2))*(self.final_size/crop_aug_h)

            print_flag = False
            totaltemprand = np.random.rand()
            if totaltemprand < self.total_ratio:
                temprand = np.random.rand()
                if self.sunglass > 0 and temprand < self.sunglass_ratio:
                    # radious = 30 + np.random.rand()*15
                    radious = self.final_size/2.0/4.0 + np.random.rand()*(self.final_size/2.0/4.0/2.0)
                    cv2.circle(img,(int(LEx_mod),int(LEy_mod)),int(radious),(0,0,0),-1)
                    cv2.circle(img,(int(REx_mod),int(REy_mod)),int(radious),(0,0,0),-1)
                    if print_flag:
                        print ("sunglass")
                        cv2.imwrite("sunglass3.jpg",img)

                elif self.hat>0 and (temprand - self.sunglass_ratio) < self.hat_ratio:
                    hat_w = (REx_mod - LEx_mod)*2.4
                    hat_h = (LEy_mod)*(0.7+(1.2-0.7)*np.random.rand())
                    hat_l = max(int((REx_mod + LEx_mod)/2 - hat_w/2),1)
                    hat_t = 1
                    hat_r = min(int(hat_l + hat_w),(self.final_size - 1))
                    hat_b = min(int(hat_t + hat_h),(self.final_size - 1))
                    for i in range(hat_t,hat_b+1):
                        for j in range(hat_l,hat_r+1):
                            img[i,j][0] = np.random.randint(0,256)
                            img[i,j][1] = np.random.randint(0,256)
                            img[i,j][2] = np.random.randint(0,256)
                    if print_flag:
                        print("hat")
                        cv2.imwrite("hat3.jpg",img)

                elif self.mask > 0 :
                    mask_w = (REx_mod - LEx_mod)*2.4
                    mask_h = (self.final_size - My_mod)*(1.6+(2-1.6)*np.random.rand())
                    mask_l = max(int((REx_mod + LEx_mod)/2 - mask_w/2),1)
                    mask_r = min(int(mask_l+mask_w),(self.final_size - 1))
                    mask_t = max(int(self.final_size  - mask_h),1)
                    mask_b = self.final_size - 1
                    for i in range(mask_t,mask_b+1):
                        for j in range(mask_l,mask_r + 1):
                            img[i,j][0] = np.random.randint(0,256)
                            img[i,j][1] = np.random.randint(0,256)
                            img[i,j][2] = np.random.randint(0,256)
                    if print_flag:
                        print("mask")
                        cv2.imwrite("mask3.jpg",img)
            else:
                if print_flag:
                    print("Nothing")
        return img


class FaceAugmentationCV2Template(object):
    def __init__(self, crop_size, final_size, crop_center_y_offset, scale_aug, trans_aug, mask_spec,
                total_ratio,sunglass,sunglass_ratio,mask,mask_ratio,hat,hat_ratio,flip=-1,mask_type='template'):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip
        #####add mask_spec by wl
        self.mask_spec = mask_spec
        self.total_ratio = total_ratio
        self.sunglass = sunglass
        self.sunglass_ratio = sunglass_ratio
        self.mask = mask
        self.mask_ratio = mask_ratio
        self.hat = hat
        self.hat_ratio = hat_ratio

    def __call__(self, img):
        is_sunglass = False
        is_hat = False
        is_mask = False
        print_flag = False

        if self.mask_spec == True:
            LEx = 70.7
            LEy = 113.0
            REx = 108.23
            REy = 113.0
            Mx = 89.43
            My = 153.51

            totaltemprand = np.random.rand()
            if totaltemprand < self.total_ratio:
                temprand = np.random.rand()

                workpath = os.path.abspath('.')

                if self.sunglass > 0.0 and temprand <= self.sunglass_ratio:
                    radious = 15 + np.random.rand()*15
                    #radious = self.final_size/2.0/4.0 + np.random.rand()*(self.final_size/2.0/4.0/2.0)
                    cv2.circle(img,(int(LEx),int(LEy)),int(radious),(0,0,0),-1)
                    cv2.circle(img,(int(REx),int(REy)),int(radious),(0,0,0),-1)
                    if print_flag:
                        # cv2.imwrite('sunglasses.jpg',img)
                        is_sunglass = True

                elif self.hat>0.0 and (temprand - self.sunglass_ratio) <= self.hat_ratio:
                    dirpath = os.path.join(workpath, 'mask_templates/hat/')
                    hatpaths = os.listdir(dirpath)
                    hatpath = dirpath + random.sample(hatpaths,1)[0]
                    hat = Image.open(hatpath)
                    t_width = hat.width
                    t_height = hat.height
                    totalx = 0.0
                    totaly = 0.0
                    count = 1
                    r,g,b,alpha = hat.split()
                    for y in range(t_height):
                        for x in range(t_width):
                            pixel = alpha.getpixel((x,y))
                            if pixel > 0:
                                totalx = totalx + x
                                totaly = totaly + y
                                count = count + 1
                    avrx = int(totalx/count)
                    avry = int(totaly/count)
                    gap = 89 - avrx
                    xstart = int(0 + gap)
                    ystart = int(np.random.rand()*20)
                    xend = xstart + 178
                    yend = ystart + 218
                    tmpimg = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                    tmpimg.paste(hat,(xstart,ystart,xend,yend),mask=alpha)
                    img = cv2.cvtColor(np.array(tmpimg),cv2.COLOR_RGB2BGR)
                    if print_flag:
                        # cv2.imwrite('hat.jpg',img)
                        is_hat = True

                elif self.mask > 0.0 :
                    dirpath = os.path.join(workpath, 'mask_templates/respirator/')
                    maskpaths = os.listdir(dirpath)
                    maskpath = dirpath + random.sample(maskpaths,1)[0]
                    mask = Image.open(maskpath)
                    t_width = mask.width
                    t_height = mask.height
                    totalx = 0.0
                    totaly = 0.0
                    count = 1
                    r,g,b,alpha = mask.split()
                    for y in range(t_height):
                        for x in range(t_width):
                            pixel = alpha.getpixel((x,y))
                            if pixel > 0:
                                totalx = totalx + x
                                totaly = totaly + y
                                count = count + 1
                    avrx = int(totalx/count)
                    avry = int(totaly/count)
                    gap = 89-avrx
                    xstart = int(0 + gap)
                    ystart = int(153-avry)
                    xend = xstart + 178
                    yend = ystart + 218
                    tmpimg = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                    tmpimg.paste(mask,(xstart,ystart,xend,yend),mask=alpha)
                    img = cv2.cvtColor(np.array(tmpimg),cv2.COLOR_RGB2BGR)
                    if print_flag:
                        # cv2.imwrite('mask.jpg',img)
                        is_mask = True

                else:
                    pdb.set_trace()

        ## transform
        scale_diff_h = (np.random.rand()*2-1)*self.scale_aug
        scale_diff_w = (np.random.rand()*2-1)*self.scale_aug
        crop_aug_h = self.crop_size*(1+scale_diff_h)
        crop_aug_w = self.crop_size*(1+scale_diff_w)

        trans_diff_h = (np.random.rand()*2-1)*self.trans_aug
        trans_diff_w = (np.random.rand()*2-1)*self.trans_aug

        h, w, _ = img.shape
        ct_x = w/2*(1+trans_diff_w)
        ct_y = (h/2+self.crop_center_y_offset)*(1+trans_diff_h)

        if ct_x < crop_aug_w/2:
            crop_aug_w = ct_x*2 - 0.5
        if ct_y < crop_aug_h/2:
            crop_aug_h = ct_y*2 - 0.5
        if ct_x + crop_aug_w/2 >= w:
            crop_aug_w = (w-ct_x)*2 - 0.5
        if ct_y + crop_aug_h/2 >= h:
            crop_aug_h = (h-ct_y)*2 - 0.5

        #rect = (ct_x-crop_aug_w/2, ct_y-crop_aug_h/2, ct_x+crop_aug_w/2, ct_y+crop_aug_h/2)
        #img = img.resize((self.final_size, self.final_size), box=rect)
        t = int(np.ceil(ct_y-crop_aug_h/2))
        #d = int(np.ceil(ct_y+crop_aug_h/2))
        l = int(np.ceil(ct_x-crop_aug_w/2))
        #r = int(np.ceil(ct_x+crop_aug_w/2))
        img = img[t:int(t+crop_aug_h),l:int(l+crop_aug_w),:]
        img = cv2.resize(img, (self.final_size, self.final_size))

        if print_flag:
            if is_sunglass:
                cv2.imwrite('sunglass.jpg',img)
            elif is_hat:
                cv2.imwrite('hat.jpg',img)
            elif is_mask:
                cv2.imwrite('mask.jpg',img)

        #print(self.flip)
        if np.random.rand() <= self.flip:
            #print('do flip')
            img = cv2.flip(img, 1)

        return img
