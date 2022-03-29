import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


class DatasetPSSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetPSSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/M/H
        # ------------------------------------
        self.paths_H = sorted(util.get_image_paths(opt['dataroot_H']))
        self.paths_M = sorted(util.get_image_paths(opt['dataroot_M'])) if opt['dataroot_M'] is not None else None
        self.paths_L = sorted(util.get_image_paths(opt['dataroot_L']))

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_M and self.paths_H:
            assert len(self.paths_L) == len(self.paths_M) == len(self.paths_H), 'L/M/H mismatch - {}, {}, {}.'.format(len(self.paths_L), len(self.paths_M), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        M_path = None

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_Left_path = self.paths_H[2 * index]
        img_H_Left = util.imread_uint(H_Left_path, self.n_channels)
        img_H_Left = util.uint2single(img_H_Left)

        H_Right_path = self.paths_H[2 * index + 1]
        img_H_Right = util.imread_uint(H_Right_path, self.n_channels)
        img_H_Right = util.uint2single(img_H_Right)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H_Left = util.modcrop(img_H_Left, self.sf)
        img_H_Right = util.modcrop(img_H_Right, self.sf)

        # ------------------------------------
        # get M image
        # ------------------------------------
        if self.paths_M:
            # --------------------------------
            # directly load M image
            # --------------------------------
            M_Left_path = self.paths_M[2 * index]
            img_M_Left = util.imread_uint(M_Left_path, self.n_channels)
            img_M_Left = util.uint2single(img_M_Left)

            M_Right_path = self.paths_M[2 * index + 1]
            img_M_Right = util.imread_uint(M_Right_path, self.n_channels)
            img_M_Right = util.uint2single(img_M_Right)

        else:
            # --------------------------------
            # sythesize M image via matlab's bicubic
            # --------------------------------
            H, W = img_H_Left.shape[:2]
            img_M_Left = util.imresize_np(img_H_Left, 1 / self.sf * 2, True)
            img_M_Right = util.imresize_np(img_H_Right, 1 / self.sf * 2, True)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_Left_path = self.paths_L[2 * index]
            img_L_Left = util.imread_uint(L_Left_path, self.n_channels)
            img_L_Left = util.uint2single(img_L_Left)

            L_Right_path = self.paths_L[2 * index + 1]
            img_L_Right = util.imread_uint(L_Right_path, self.n_channels)
            img_L_Right = util.uint2single(img_L_Right)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H_Left.shape[:2]
            img_L_Left = util.imresize_np(img_H_Left, 1 / self.sf, True)
            img_L_Right = util.imresize_np(img_H_Right, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/M/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L_Left.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L_Left = img_L_Left[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
            img_L_Right = img_L_Right[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding M patch
            # --------------------------------
            rnd_h_M, rnd_w_M = int(rnd_h * self.sf // 2), int(rnd_w * self.sf // 2)
            img_M_Left = img_M_Left[rnd_h_M:rnd_h_M + self.patch_size // 2, rnd_w_M:rnd_w_M + self.patch_size // 2, :]
            img_M_Right = img_M_Right[rnd_h_M:rnd_h_M + self.patch_size // 2, rnd_w_M:rnd_w_M + self.patch_size // 2, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H_Left = img_H_Left[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
            img_H_Right = img_H_Right[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = 2 * random.randint(0, 3)
            img_L_Left, img_M_Left, img_H_Left = util.augment_img(img_L_Left, mode=mode), util.augment_img(img_M_Left, mode=mode), util.augment_img(img_H_Left, mode=mode)
            img_L_Right, img_M_Right, img_H_Right = util.augment_img(img_L_Right, mode=mode), util.augment_img(img_M_Right, mode=mode), util.augment_img(img_H_Right, mode=mode)

            # RGB Prem
            perm = np.random.permutation(3)
            img_L_Left, img_M_Left, img_H_Left = img_L_Left[:,:,perm], img_M_Left[:,:,perm], img_H_Left[:,:,perm]
            img_L_Right, img_M_Right, img_H_Right = img_L_Right[:,:,perm], img_M_Right[:,:,perm], img_H_Right[:,:,perm]

        # ---------------------------------------
        # L/M/H pairs, HWC to CHW, numpy to tensor
        # ---------------------------------------
        img_H_Left, img_M_Left, img_L_Left = util.single2tensor3(img_H_Left), util.single2tensor3(img_M_Left), util.single2tensor3(img_L_Left)
        img_H_Right, img_M_Right, img_L_Right = util.single2tensor3(img_H_Right), util.single2tensor3(img_M_Right), util.single2tensor3(img_L_Right)
        
        return {'L_Left': img_L_Left, 'L_Right': img_L_Right,
                'M_Left': img_M_Left, 'M_Right': img_M_Right,
                'H_Left': img_H_Left, 'H_Right': img_H_Right,
                'L_Left_path': L_Left_path, 'L_Right_path': L_Right_path, 
                'M_Left_path': L_Left_path, 'M_Right_path': L_Right_path,
                'H_Left_path': H_Left_path, 'H_Right_path': H_Right_path}

    def __len__(self):
        return len(self.paths_H) // 2
