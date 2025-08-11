from torch.utils.data import Dataset
import os
import cv2
import warnings
from tqdm import tqdm
import rawpy
import imageio
import numpy as np
import pickle as pkl
import torch


def scale_up(img):
    return np.uint8(img * 255.0)


def bayer2raw(packed_raw, wp=16383, bl=512):
    if torch.is_tensor(packed_raw):
        packed_raw = packed_raw.detach()
        packed_raw = packed_raw[0].cpu().float().numpy()
    packed_raw = np.clip(packed_raw, 0, 1)
    packed_raw = packed_raw * (wp - bl) + bl
    H, W, C = packed_raw.shape
    H *= 2
    W *= 2
    raw = np.empty((H, W), dtype=np.uint16)
    raw[0:H:2, 0:W:2] = packed_raw[:, :, 0]
    raw[0:H:2, 1:W:2] = packed_raw[:, :, 1]
    raw[1:H:2, 0:W:2] = packed_raw[:, :, 2]
    raw[1:H:2, 1:W:2] = packed_raw[:, :, 3]
    return raw


def raw2rgb_rawpy(packed_raw, wb=None, ccm=None, crop_x=0, crop_y=0, crop_size=512):
    """Raw2RGB pipeline (rawpy postprocess version)"""
    if packed_raw.shape[0] > 1500:
        raw = rawpy.imread("templet.dng")
        wp = 1023
        bl = 64
    else:
        raw = rawpy.imread("templet.ARW")
        wp = 16383
        bl = 512
    if wb is None:
        wb = np.array(raw.camera_whitebalance)
        wb /= wb[1]
    wb = list(wb)
    if ccm is None:
        try:
            ccm = raw.rgb_camera_matrix[:3, :3]
        except:
            warnings.warn("You have no Wei Kaixuan's customized rawpy, you can't get right ccm of SonyA7S2...")
            ccm = raw.color_matrix[:3, :3]
    elif np.max(np.abs(ccm - np.identity(3))) == 0:
        ccm = np.array(
            [
                [1.9712269, -0.6789218, -0.29230508],
                [-0.29104823, 1.748401, -0.45735288],
                [0.02051281, -0.5380369, 1.5175241],
            ],
            dtype=np.float32,
        )

    if len(packed_raw.shape) >= 3:
        raw.raw_image_visible[crop_x:crop_x + crop_size * 2, crop_y:crop_y + crop_size * 2] = bayer2raw(packed_raw, wp, bl)
    else:  # 传进来的就是raw图
        raw.raw_image_visible[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size] = packed_raw
    
    data = cv2.resize(
        data, (data.shape[1] // 2, data.shape[0] // 2), interpolation=cv2.INTER_AREA
    )

    out = raw.postprocess(
        use_camera_wb=False,
        user_wb=wb,
        half_size=True,
        no_auto_bright=True,
        output_bps=8,
        bright=1,
        user_black=None,
        user_sat=None,
    )

    return out[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size] / 255


class RAWImageDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        file_list,
        raw_min_value=255,
        raw_max_value=16383,
        transforms=None,
        rgb_only=False,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.file_list = os.path.join(dataset_path, file_list)
        self.rgb_only = rgb_only

        self.data = self.load()
        self.transforms = transforms

        self.raw_min_value = raw_min_value
        self.raw_max_value = raw_max_value

    def load(self):
        data = []

        with open(self.file_list, "r") as f_read:
            item_list = [line.strip() for line in f_read.readlines()]

        for item in item_list:
            parts = item.split(",")
            assert len(parts) == 2, f"invalid item: {item}"
            raw_rel_path, rgb_rel_path = parts

            raw_path = os.path.join(self.dataset_path, raw_rel_path)
            rgb_path = os.path.join(self.dataset_path, rgb_rel_path)

            if (self.rgb_only or os.path.exists(raw_path)) and os.path.exists(rgb_path):
                data.append((raw_path, rgb_path))
            else:
                print(f"Warning: {raw_path} or {rgb_path} does not exist")

        return data

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2, 0, 1)

    def __getitem__(self, idx: int):
        raw_path, rgb_path = self.data[idx]

        if os.path.splitext(rgb_path)[1] == ".npy":
            rgb_data = np.load(rgb_path)
        else:
            rgb_data = imageio.imread(rgb_path)

        rgb_data = rgb_data.astype(np.float32) / 255

        if not self.rgb_only:
            raw_data_np = np.load(raw_path)
            raw_data = raw_data_np["raw"]
            raw_data = (raw_data.astype(np.float32) - self.raw_min_value) / (
                self.raw_max_value - self.raw_min_value
            )
        else:
            raw_data = np.zeros(
                (rgb_data.shape[0], rgb_data.shape[1], 4), dtype=np.float32
            )

        if rgb_data.shape[:2] != raw_data.shape[:2]:
            raise ValueError(
                f"target_rgb_img.shape: {rgb_data.shape}, input_raw_img.shape: {raw_data.shape}, file_name: {raw_path}, {rgb_path}"
            )

        if self.transforms is not None:
            raw_data, rgb_data = self.transforms(raw_data, rgb_data)

        raw_data = np.clip(raw_data, 0, 1)

        raw_data = self.np2tensor(raw_data).float()
        rgb_data = self.np2tensor(rgb_data).float()

        raw_data = raw_data * 2 - 1
        rgb_data = rgb_data * 2 - 1

        out_dict = {
            "raw_data": raw_data,
            "guidance_data": rgb_data,
            "path": os.path.relpath(rgb_path, self.dataset_path),
        }

        return out_dict

    def __len__(self):
        return len(self.data)


# class SIDRAWImageDataset(Dataset):
#     def __init__(
#         self,
#         dataset_path,
#         file_list,
#         raw_min_value=512,
#         raw_max_value=16383,
#         transforms=None,
#         rgb_only=False,
#     ) -> None:
#         super().__init__()

#         self.dataset_path = dataset_path
#         self.file_list = os.path.join(dataset_path, file_list)
#         self.rgb_only = rgb_only

#         self.data = self.load()
#         self.transforms = transforms

#         self.raw_min_value = raw_min_value
#         self.raw_max_value = raw_max_value

#     def load(self):
#         data = []

#         with open(self.file_list, "r") as f_read:
#             item_list = [line.strip() for line in f_read.readlines()]

#         for item in item_list:
#             # parts = item.split(",")
#             # assert len(parts) == 2, f"invalid item: {item}"
#             raw_rel_path = item

#             raw_path = os.path.join(self.dataset_path, raw_rel_path)

#             if self.rgb_only or os.path.exists(raw_path):
#                 data.append(raw_path)
#             else:
#                 print(f"Warning: {raw_path} does not exist")

#         return data

#     def np2tensor(self, array):
#         return torch.FloatTensor(array)

#     def __getitem__(self, idx: int):
#         raw_path = self.data[idx]

#         raw_info = np.load(raw_path)
#         raw_data_np = raw_info['im']

#         if rgb_data.shape[:2] != raw_data.shape[:2]:
#             raise ValueError(
#                 f"target_rgb_img.shape: {rgb_data.shape}, input_raw_img.shape: {raw_data.shape}, file_name: {raw_path}, {rgb_path}"
#             )

#         if self.transforms is not None:
#             raw_data, rgb_data = self.transforms(raw_data, rgb_data)

#         raw_data = np.clip(raw_data, 0, 1)

#         raw_data = self.np2tensor(raw_data).float()
#         rgb_data = self.np2tensor(rgb_data).float()

#         raw_data = raw_data * 2 - 1
#         rgb_data = rgb_data * 2 - 1

#         out_dict = {
#             "raw_data": raw_data,
#             "guidance_data": rgb_data,
#             "path": os.path.relpath(rgb_path, self.dataset_path),
#         }

#         return out_dict

#     def __len__(self):
#         return len(self.data)
    

# class SID_Dataset(Dataset):
#     def __init__(self, transforms=None, train=True, wl=16383, bl=512, clip_low=False, clip_high=True):
#         super().__init__()
#         self.transforms = transforms
#         self.wl, self.bl = wl, bl
#         self.clip_low = 0 if clip_low else float("-inf")
#         self.clip_high = 1 if clip_high else float("inf")

#         ## load pmn's darkshading
#         with open(f"/root/WorkSpace/raw_image_denoising-main/resources/SonyA7S2/darkshading_BLE.pkl", "rb") as f:
#             self.pmn_ble = pkl.load(f)
#         self.pmn_dsk_high = np.load(f"/root/WorkSpace/raw_image_denoising-main/resources/SonyA7S2/darkshading_highISO_k.npy")
#         self.pmn_dsk_low = np.load(f"/root/WorkSpace/raw_image_denoising-main/resources/SonyA7S2/darkshading_lowISO_k.npy")
#         self.pmn_dsb_high = np.load(f"/root/WorkSpace/raw_image_denoising-main/resources/SonyA7S2/darkshading_highISO_b.npy")
#         self.pmn_dsb_low = np.load(f"/root/WorkSpace/raw_image_denoising-main/resources/SonyA7S2/darkshading_lowISO_b.npy")

#         ## format data pairs
#         if train:
#             with open(f"./infos/SID_train.info", "rb") as info_file:
#                 self.data_info = pkl.load(info_file)
#         else:
#             with open(f"./infos/SID_test.info", "rb") as info_file:
#                 self.data_info = pkl.load(info_file)

#         self.cache = {}
#         for idx in tqdm(range(len(self.data_info))):
#             hr_raw = np.array(rawpy.imread(self.data_info[idx]["long"]).raw_image_visible).astype(np.float32)
#             self.cache[idx] = hr_raw

#     def __len__(self):
#         return len(self.data_info)

#     def get_darkshading(self, iso):
#         if iso <= 1600:
#             return self.pmn_dsk_low * iso + self.pmn_dsb_low + self.pmn_ble[iso]
#         else:
#             return self.pmn_dsk_high * iso + self.pmn_dsb_high + self.pmn_ble[iso]

#     def pack_raw(self, img, norm=False, clip=False):
#         out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
#         out = (out - self.bl) / (self.wl - self.bl) if norm else out
#         out = np.clip(out, 0, 1) if clip else out
#         return out.astype(np.float32)

#     def np2tensor(self, array):
#         return torch.FloatTensor(array).permute(2, 0, 1)
    
#     def __getitem__(self, idx):
#         ## load data
#         hr_raw = self.cache[idx]

#         ## subtract dark shading
#         hr_raw = hr_raw - self.get_darkshading(self.data_info[idx]["ISO"])

#         ## pack to 4-chans
#         raw_data = self.pack_raw(hr_raw, norm=True, clip=True)  ## [h, w, c]

#         rgb_data = raw2rgb_rawpy(np.clip(raw_data, 0, 1), wb=self.data_info[idx]['wb'], ccm=self.data_info[idx]['ccm'])

#         if self.transforms is not None:
#             raw_data, rgb_data = self.transforms(raw_data, rgb_data)

#         raw_data = np.clip(raw_data, 0, 1)
#         raw_data = self.np2tensor(raw_data)
#         rgb_data = self.np2tensor(rgb_data)

#         raw_data = raw_data * 2 - 1
#         rgb_data = rgb_data * 2 - 1

#         data = {
#             "raw_data": raw_data,
#             "guidance_data": rgb_data,
#             "path": f"{self.data_info[idx]['name'][:5]}_00_{int(self.data_info[idx]['ExposureTime'])}s",
#         }

#         return data


class SID_Dataset(Dataset):
    def __init__(self, transforms=None, train=True, wl=16383, bl=512, clip_low=False, clip_high=True):
        super().__init__()
        self.transforms = transforms
        self.wl, self.bl = wl, bl
        self.clip_low = 0 if clip_low else float("-inf")
        self.clip_high = 1 if clip_high else float("inf")
        self.data_raw_dir = "/root/WorkSpace/DataSet/SID/Sony_train_long_patches"
        self.data_rgb_dir = "/root/WorkSpace/DataSet/SID/Sony/long_rgb"

        ## load pmn's darkshading
        with open(f"/root/WorkSpace/CVPR25/resources/SonyA7S2/darkshading_BLE.pkl", "rb") as f:
            self.pmn_ble = pkl.load(f)
        self.pmn_dsk_high = np.load(f"/root/WorkSpace/CVPR25/resources/SonyA7S2/darkshading_highISO_k.npy")
        self.pmn_dsk_low = np.load(f"/root/WorkSpace/CVPR25/resources/SonyA7S2/darkshading_lowISO_k.npy")
        self.pmn_dsb_high = np.load(f"/root/WorkSpace/CVPR25/resources/SonyA7S2/darkshading_highISO_b.npy")
        self.pmn_dsb_low = np.load(f"/root/WorkSpace/CVPR25/resources/SonyA7S2/darkshading_lowISO_b.npy")

        ## format data pairs
        if train:
            with open(f"./data/SID/Sony_train_long.txt", "r") as info_file:
                self.data_info = info_file.read().splitlines()
        else:
            with open(f"./data/SID/Sony_test_long.txt", "r") as info_file:
                self.data_info = info_file.read().splitlines()

        self.cache = {}
        for idx in tqdm(range(len(self.data_info))):
            self.cache[idx]= np.load(os.path.join(self.data_raw_dir, self.data_info[idx]))

    def __len__(self):
        return len(self.data_info)

    def get_darkshading(self, iso):
        if iso <= 1600:
            return self.pmn_dsk_low * iso + self.pmn_dsb_low + self.pmn_ble[iso]
        else:
            return self.pmn_dsk_high * iso + self.pmn_dsb_high + self.pmn_ble[iso]

    def pack_raw(self, img, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=0)
        out = (out - self.bl) / (self.wl - self.bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def np2tensor(self, array):
        return torch.FloatTensor(array).permute(2, 0, 1)
    
    def __getitem__(self, idx):
        ## load data
        name = self.data_info[idx][:12] + '.npy'
        hr_raw = self.cache[idx]['im']

        ## subtract dark shading
        c, h, w = hr_raw.shape
        x, y = int(self.cache[idx]['crop_x']), int(self.cache[idx]['crop_y'])
        hr_raw = hr_raw - self.pack_raw(self.get_darkshading(int(self.cache[idx]["iso"])))[:, x:x + h, y:y + w]
        hr_raw = (hr_raw - self.bl) / (self.wl - self.bl)
        hr_raw = np.clip(hr_raw, 0, 1)

        rgb_data = np.load(os.path.join(self.data_rgb_dir, name)) / 255
        rgb_data = rgb_data[x:x + h, y:y + w, :]
        raw_data = hr_raw.transpose(1, 2, 0)

        if self.transforms is not None:
            raw_data, rgb_data = self.transforms(raw_data, rgb_data)

        raw_data = self.np2tensor(raw_data)
        rgb_data = self.np2tensor(rgb_data)

        raw_data = raw_data * 2 - 1
        rgb_data = rgb_data * 2 - 1

        data = {
            "raw_data": raw_data,
            "guidance_data": rgb_data,
            "path": f"{os.path.splitext(self.data_info[idx])[0]}",
        }

        return data