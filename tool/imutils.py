import PIL.Image
import random
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic


def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x[x != x_max] = 0
    return x


#This is superpixel_strategy
def hide_patch_ss(img):

    _,w,h = img.shape
    uncontribute_num = 0
    contribute_num = 0
    # You can value n_segments to set the number of super-pixels
    spx = slic(img.permute(1,2,0),compactness=10,n_segments=300,start_label = 1)
    sign = torch.ones((img.shape))
    patch_num = np.random.randint(0,np.max(spx)+1,np.floor((np.max(spx)+1)*0.5).astype(np.int))
    dic = {}
    for num in patch_num:
        dic[num] = 0
    for i in range(w):
        for j in range(h):
            if spx[i,j] in dic:
                sign[:, i, j] = 0
                uncontribute_num += 1
            else:
                contribute_num += 1



    img1 = img * sign
    img2 = img * (1-sign)
    contri_rate= contribute_num / (contribute_num + uncontribute_num)
    uncontri_rate = 1-contri_rate

    return img1,img2,contri_rate,uncontri_rate



#This is grid_strategy
def hide_patch(img):

    s = img.shape
    wd = s[1]
    ht = s[2]
    uncontribute_num = 0
    contribute_num = 0
    # possible grid size, which is actually a set
    grid_sizes = [224, 448]
    sign = torch.ones((s))
    # hiding probability
    hide_prob = 0.5

    # randomly choose one grid size
    grid_size = grid_sizes[random.randint(0, len(grid_sizes) - 1)]

    # hide the patches
    if (grid_size != 0):
        for x in range(0, wd, grid_size):
            for y in range(0, ht, grid_size):
                x_end = min(wd, x + grid_size)
                y_end = min(ht, y + grid_size)
                if (random.random() <= hide_prob):
                    sign[:,x:x_end,y:y_end] = 0
                    uncontribute_num += 1
                else:
                    contribute_num += 1

    img1 = img * sign
    img2 = img * (1-sign)
    contri_rate= contribute_num / (contribute_num + uncontribute_num)
    uncontri_rate = 1-contri_rate
    return img1,img2,contri_rate,uncontri_rate


class RandomResizeLong():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img):

        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        img = img.resize(target_shape, resample=PIL.Image.CUBIC)

        return img


class RandomCrop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]

        return container

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def crop_with_box(img, box):
    if len(img.shape) == 3:
        img_cont = np.zeros((max(box[1]-box[0], box[4]-box[5]), max(box[3]-box[2], box[7]-box[6]), img.shape[-1]), dtype=img.dtype)
    else:
        img_cont = np.zeros((max(box[1] - box[0], box[4] - box[5]), max(box[3] - box[2], box[7] - box[6])), dtype=img.dtype)
    img_cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return img_cont


def random_crop(images, cropsize, fills):
    if isinstance(images[0], PIL.Image.Image):
        imgsize = images[0].size[::-1]
    else:
        imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, fills):

        if isinstance(img, PIL.Image.Image):
            img = img.crop((box[6], box[4], box[7], box[5]))
            cont = PIL.Image.new(img.mode, (cropsize, cropsize))
            cont.paste(img, (box[2], box[0]))
            new_images.append(cont)

        else:
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            new_images.append(cont)

    return new_images


class AvgPool2d():

    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, img):
        import skimage.measure

        return skimage.measure.block_reduce(img, (self.ksize, self.ksize, 1), np.mean)


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


class RescaleNearest():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, npimg):
        import cv2
        return cv2.resize(npimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)




def crf_inference(img, probs, t=5, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=50/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def max_norm(p, version='torch', e=1e-5):
    if version is 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
            min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
        elif p.dim() == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
    elif version is 'numpy' or version is 'np':
        if p.ndim == 3:
            C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(1,2),keepdims=True)
            min_v = np.min(p,(1,2),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(2,3),keepdims=True)
            min_v = np.min(p,(2,3),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
    return p


# def hide_patch_pixel(img):
#     # get width and height of the image
#     s = img.shape
#     wd = s[1]
#     ht = s[2]
#     sign = np.random.randint(0,2,(wd,ht))
#
#     img1 = img * sign
#     img2 = img * (1-sign)
#
#     return img1,img2,0.5,0.5