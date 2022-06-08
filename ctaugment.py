'''
# author of this script: @shreejalt
# original author / copyright: https://github.com/google-research/remixmatch/blob/master/libml/ctaugment.py
# This script is taken from the above given link. Have done the changes to support the wrapper function of CTAugment
# and weight update.

'''
import json
import random
from collections import namedtuple, OrderedDict
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torchvision

OP = namedtuple("OP", ("f", "bins"))


class InfiniteDataLoader(torch.utils.data.DataLoader): 
    # Taken from https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class ProbeDataset(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        transform,
    ):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        output = {
            'label': target,
        }
        
        for transform in self.transform.transforms:
            if isinstance(transform, CTAugment):
                img, augs = transform.__probe__(img)
                output['augs'] = json.dumps(augs)
            else:
                img = transform(img)
        output['img'] = img

        return output


class CTUpdater(object):

    def __init__(
        self,
        datapath,
        train_transforms,
        batch_size=64,
        num_workers=4,
        gpu=None
    ):
        # Necessary arguments
        self.datapath = datapath
        self.train_transforms = train_transforms
        self.CTClass = None

        for obj in self.train_transforms.transforms:
            if isinstance(obj, CTAugment):
                self.CTClass = obj
                break
        if self.CTClass is None:
            raise ValueError('CTAugment instance is not present in transforms')

        # Optional arguments
        self.gpu = gpu

        # Initializing Probe Dataset
        probe_dataset = ProbeDataset(datapath, train_transforms)
        self.probe_loader = InfiniteDataLoader(
            probe_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        self.probe_loader_iter = iter(self.probe_loader)

    def update(self, model):

        batch = next(self.probe_loader_iter)
        
        model.eval()

        images, labels, policies = self.parse_batch(batch)

        with torch.no_grad():
            y_pred = model(images)
            y_probs = torch.softmax(y_pred, dim=1)
            error_per_op = list()

            for y_prob, t, policy in zip(y_probs, labels, policies):
                error = y_prob
                error[t] -= 1
                error = torch.abs(error).sum()

                error_per_op.append([policy, error])

            for t in error_per_op:
                policy, error = t
                self.CTClass.update_rates(policy, 1.0 - 0.5 * error)

        model.train()

    def parse_batch(self, batch):

        def deserialize(policy):
            return [OP(f=x[0], bins=x[1]) for x in json.loads(policy)]

        input = batch['img']
        label = batch['label']
        policies = [deserialize(policy_str) for policy_str in batch['augs']]

        if self.gpu is not None:
            input = input.cuda(self.gpu)
            label = label.cuda(self.gpu)

        return input, label, policies


class CTAugment(object):

    def __init__(
        self,
        depth=2,
        decay=0.999,
        thresh=0.8,
        geo_augs=False,
        color_augs=False,
    ):
        # Optional arguments
        self.depth = depth
        self.decay = decay
        self.thresh = thresh

        assert(not geo_augs and not color_augs)

        self.rates, self.OPS = dict(), dict()

        augs = get_all_augs()
        if geo_augs:
            augs = get_geo_augs()
        if color_augs:
            augs = get_color_augs()

        self.augs = augs

        for aug in augs:
            func, bins = aug[0], tuple(aug[1:])
            self.OPS[func.__name__] = OP(func, bins)
            self.rates[func.__name__] = tuple([np.ones(x, "f") for x in bins])

    def __call__(self, img):

        aug_list = self.get_train_augs() + [OP('cutout', (1,))]  # Add Cutout during training according to the original implementation

        if aug_list is None:
            return img

        for op, args in aug_list:
            img = self.OPS[op].f(img, *args)

        return img

    def __probe__(self, img):

        aug_list = self.get_probe_augs()
        if aug_list is None:
            return img, aug_list

        for op, args in aug_list:
            img = self.OPS[op].f(img, *args)

        return img, aug_list

    def load_state_dict(self, state):

        for k in ['decay', 'depth', 'thresh', 'rates', 'OPS', 'augs']:
            assert k in state, "{} not in {}".format(k, state.keys())
            setattr(self, k, state[k])

    def state_dict(self):

        odict = OrderedDict([(k, getattr(self, k)) for k in ['decay', 'depth', 'thresh', 'rates', 'OPS', 'augs']])
        return odict

    @property
    def stats(self):
        return "\n".join(
            "%-16s    %s"
            % (
                k,
                " / ".join(
                    " ".join("%.2f" % x for x in self.calc_mcap(rate))
                    for rate in self.rates[k]
                ),
            )
            for k in sorted(self.OPS.keys())
        )

    def get_probe_augs(self):

        list_ops = list(self.OPS.keys())

        aug_list = list()

        for _ in range(self.depth):
            choice_aug = random.choice(list_ops)
            bins = self.rates[choice_aug]
            rnd = np.random.uniform(0, 1, len(bins))
            aug_list.append(OP(choice_aug, rnd.tolist()))

        return aug_list

    def get_train_augs(self):

        list_ops = list(self.OPS.keys())
        aug_list = list()

        for _ in range(self.depth):
            per_aug = list()
            choice_aug = random.choice(list_ops)
            bins = self.rates[choice_aug]

            rnd = np.random.uniform(0, 1, len(bins))

            for r, bin in zip(rnd, bins):
                mcap = self.calc_mcap(bin)
                norm_mcap = np.random.choice(mcap.shape[0], p=mcap / mcap.sum())
                per_aug.append((norm_mcap + r) / mcap.shape[0])

            aug_list.append(OP(choice_aug, per_aug))

        return aug_list

    def calc_mcap(self, m):

        mcap = m + (1 - self.decay)
        norm_mcap = mcap / mcap.max()
        norm_mcap[norm_mcap < self.thresh] = 0
        return norm_mcap

    def update_rates(self, policy, prox):

        for k, bins in policy:
            for p, rate in zip(bins, self.rates[k]):
                p = int(p * len(rate) * 0.999)
                rate[p] = rate[p] * self.decay + prox * (1 - self.decay)


def _enhance(x, op, level):
    return op(x).enhance(0.1 + 1.9 * level)


def _imageop(x, op, level):
    return Image.blend(x, op(x), level)


def _filter(x, op, level):
    return Image.blend(x, x.filter(op), level)


def get_all_augs():

    list_augs = [
        (autocontrast, 17),
        (blur, 17),
        (brightness, 17),
        (color, 17),
        (contrast, 17),
        (cutout, 17),
        (equalize, 17),
        (invert, 17),
        (identity, ),
        (posterize, 8),
        (rescale, 17, 6),
        (rotate, 17),
        (sharpness, 17),
        (shear_x, 17),
        (shear_y, 17),
        (smooth, 17),
        (solarize, 17),
        (translate_x, 17),
        (translate_y, 17)
    ]
    return list_augs


def get_geo_augs():

    list_augs = [
        (cutout, 17),
        (identity, ),
        (posterize, 8),
        (rescale, 17, 6),
        (rotate, 17),
        (sharpness, 17),
        (shear_x, 17),
        (shear_y, 17),
        (translate_x, 17),
        (translate_y, 17)
    ]
    return list_augs


def get_color_augs():

    list_augs = [
        (autocontrast, 17),
        (blur, 17),
        (brightness, 17),
        (color, 17),
        (contrast, 17),
        (cutout, 17),
        (equalize, 17),
        (invert, 17),
        (identity, ),
        (posterize, 8),
        (smooth, 17),
        (solarize, 17),
    ]
    return list_augs


def autocontrast(x, level):
    return _imageop(x, ImageOps.autocontrast, level)


def blur(x, level):
    return _filter(x, ImageFilter.BLUR, level)


def brightness(x, brightness):
    return _enhance(x, ImageEnhance.Brightness, brightness)


def color(x, color):
    return _enhance(x, ImageEnhance.Color, color)


def contrast(x, contrast):
    return _enhance(x, ImageEnhance.Contrast, contrast)


def cutout(x, level):

    """Apply cutout to pil_img at the specified level."""
    size = 1 + int(level * min(x.size) * 0.499)
    img_height, img_width = x.size
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (
        min(img_height, height_loc + size // 2),
        min(img_width, width_loc + size // 2),
    )
    pixels = x.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        for j in range(upper_coord[1], lower_coord[1]):  # For every row
            pixels[i, j] = (127, 127, 127)  # set the color accordingly
    return x


def equalize(x, level):
    return _imageop(x, ImageOps.equalize, level)


def invert(x, level):
    return _imageop(x, ImageOps.invert, level)


def identity(x):
    return x


def posterize(x, level):
    level = 1 + int(level * 7.999)
    return ImageOps.posterize(x, level)


def rescale(x, scale, method):
    s = x.size
    scale *= 0.25
    crop = (scale * s[0], scale * s[1], s[0] * (1 - scale), s[1] * (1 - scale))
    methods = (
        Image.ANTIALIAS,
        Image.BICUBIC,
        Image.BILINEAR,
        Image.BOX,
        Image.HAMMING,
        Image.NEAREST,
    )
    method = methods[int(method * 5.99)]
    return x.crop(crop).resize(x.size, method)


def rotate(x, angle):
    angle = int(np.round((2 * angle - 1) * 45))
    return x.rotate(angle)


def sharpness(x, sharpness):
    return _enhance(x, ImageEnhance.Sharpness, sharpness)


def shear_x(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


def shear_y(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


def smooth(x, level):
    return _filter(x, ImageFilter.SMOOTH, level)


def solarize(x, th):
    th = int(th * 255.999)
    return ImageOps.solarize(x, th)


def translate_x(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


def translate_y(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))