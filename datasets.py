import cv2
from torch.utils.data import Dataset
import numpy as np

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb

class CassavaDataset(Dataset):
    def __init__(self, df, TRAIN_AUGS, TEST_AUGS, mode='train', soft=False):
        super().__init__()
        self.df = df
        self.mode = mode
        self.TRAIN_AUGS = TRAIN_AUGS
        self.TEST_AUGS = TEST_AUGS
        self.soft = soft

    def __len__(self, ):
        return len(self.df)

    def to_one_hot(self, le_label, num_classes = 5):
        oho_label = np.zeros(num_classes)
        oho_label[int(le_label)] = 1
        return oho_label

    def __getitem__(self, idx):
        row = self.df[self.df.index == idx]
        image_name = row.image_id.values[0]
        img = get_img('/home/data/Cassava/train_images/' + image_name)
        if self.mode == 'train':
            label = row.label.values[0]
            if self.soft:
                label = self.to_one_hot(label)
                soft_labels = row.values[0][3:]
                label = (label * 0.7).astype(np.float16) + (soft_labels * 0.3).astype(np.float16)
            img = self.TRAIN_AUGS(image=img)['image']
            # img = np.moveaxis(img, 2, 0)
            return img, label
        elif self.mode == 'val':
            label = row.label.values[0]
            #if self.soft:
            #    soft_labels = row.values[0][3:]
            #    label = (label * 0.7).astype(np.float16) + (soft_labels * 0.3).astype(np.float16)
            img = self.TEST_AUGS(image=img)['image']
            return img, label
        elif self.mode == 'test':
            img = self.TEST_AUGS(image=img)['image']
            return img
        else:
            print("Unknown mod type")