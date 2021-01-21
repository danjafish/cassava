import cv2
from torch.utils.data import Dataset

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb

class CassavaDataset(Dataset):
    def __init__(self, df,TRAIN_AUGS, TEST_AUGS, mode='train'):
        super().__init__()
        self.df = df
        self.mode = mode
        self.TRAIN_AUGS = TRAIN_AUGS
        self.TEST_AUGS = TEST_AUGS

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[self.df.index == idx]
        image_name = row.image_id.values[0]
        img = get_img('/home/data/Cassava/train_images/' + image_name)
        if self.mode == 'train':
            label = row.label.values[0]
            img = self.TRAIN_AUGS(image=img)['image']
            # img = np.moveaxis(img, 2, 0)
            return img, label
        elif self.mode == 'val':
            label = row.label.values[0]
            img = self.VAL_AUGS(image=img)['image']
            return img, label
        elif self.mode == 'test':
            img = self.TEST_AUGS(image=img)['image']
            return img
        else:
            print("Unknown mod type")