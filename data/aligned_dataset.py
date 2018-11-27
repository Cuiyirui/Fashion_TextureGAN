import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from skimage import io
import util.util as util
from torch.autograd import Variable

#A contour image, B ground truth, C patch
class Aligned_2_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.center_crop = opt.center_crop
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize(
            (self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)

        # mid = Variable(AB, volatile=True)
        # temp = util.tensor2im(mid.data)
        # io.imshow(temp)



        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        if self.center_crop:
            w_offset = int(round((w - self.opt.fineSize) / 2.0))
            h_offset = int(round((h - self.opt.fineSize) / 2.0))
        else:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]


        if self.opt.image_initial == "VGG":
            A = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(A)
            B = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(B)
        else:
            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)


        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        if self.opt.whether_encode_cloth:
            # cliped patch as c
            C = torch.cuda.FloatTensor if self.opt.gpu_ids else torch.Tensor
            # C = torch.Tensor
            clip_start_index = (self.opt.fineSize-self.opt.encode_size)//2
            clip_end_index = clip_start_index + self.opt.encode_size
            if self.opt.which_direction == 'AtoB':
                C = B[:, clip_start_index:clip_end_index,
                                      clip_start_index:clip_end_index]
            else:
                C = A[:, clip_start_index:clip_end_index,
                    clip_start_index:clip_end_index]



        if self.opt.whether_encode_cloth:
            return {'A': A, 'B': B, 'C': C,
                    'A_paths': AB_path, 'B_paths': AB_path, 'C_paths': AB_path}
        else:
            return {'A': A, 'B': B,
                    'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'Aligned_2_Dataset'



#A contour image, B ground truth, C patch, D past patch to mask
class Aligned_3_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.center_crop = opt.center_crop
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        ABD = Image.open(AB_path).convert('RGB')
        ABD = ABD.resize(
            (self.opt.loadSize * 3, self.opt.loadSize), Image.BICUBIC)
        ABD = transforms.ToTensor()(ABD)

        # mid = Variable(AB, volatile=True)
        # temp = util.tensor2im(mid.data)
        # io.imshow(temp)



        w_total = ABD.size(2)
        w = int(w_total / 3)
        h = ABD.size(1)
        if self.center_crop:
            w_offset = int(round((w - self.opt.fineSize) / 2.0))
            h_offset = int(round((h - self.opt.fineSize) / 2.0))
        else:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = ABD[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = ABD[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]
        D = ABD[:, h_offset:h_offset + self.opt.fineSize,
               2*w + w_offset:2*w + w_offset + self.opt.fineSize]

        if self.opt.image_initial == "VGG":
            A = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(A)
            B = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(B)
        else:
            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)


        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            D = D.index_select(2, idx)

        if input_nc == 1:
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        if self.opt.whether_encode_cloth:
            # cliped patch as c
            C = torch.cuda.FloatTensor if self.opt.gpu_ids else torch.Tensor
            # C = torch.Tensor
            clip_start_index = (self.opt.fineSize-self.opt.encode_size)//2
            clip_end_index = clip_start_index + self.opt.encode_size
            if self.opt.which_direction == 'AtoB':
                C = B[:, clip_start_index:clip_end_index,
                                      clip_start_index:clip_end_index]
            else:
                C = A[:, clip_start_index:clip_end_index,
                    clip_start_index:clip_end_index]
            # join the patch
            E = self.pastePatch(C,D)


        if self.opt.whether_encode_cloth:
            return {'A': A, 'B': B, 'C': C,'E':E,
                    'A_paths': AB_path, 'B_paths': AB_path, 'C_paths': AB_path,'E_paths':AB_path}
        else:
            return {'A': A, 'B': B,
                    'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'Aligned_3_Dataset'

    def pastePatch(self,patch,mask):
        patch_num = int(self.opt.loadSize/self.opt.encode_size)

        for i in range(patch_num):
            for j in range(patch_num):
                if j==0:
                    row_im = patch
                else:
                    row_im = torch.cat((row_im,patch), 1)
            if i==0:
                final_im = row_im
            else:
                final_im = torch.cat((final_im,row_im),2)
        # denoise
        mask = mask>0.9
        final_im = torch.mul(final_im, mask.float())
        final_im = torch.add(final_im,(1-mask).float())

        return final_im