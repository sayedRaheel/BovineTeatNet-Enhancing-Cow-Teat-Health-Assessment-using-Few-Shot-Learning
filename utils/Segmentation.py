import os
import cv2
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
  def __init__(self, root_dir, transform=None,if_new_mask = False, cow_teat_binary=False):
    self.root_dir = root_dir
    self.transform = transform
    self.image_list = os.listdir(os.path.join(self.root_dir,"Images"))
    if if_new_mask: 
      self.mask_list = os.listdir(os.path.join(self.root_dir,"Masks_new"))
      self.mask_path = os.path.join(self.root_dir,"Masks_new")  
    else:
      self.mask_list = os.listdir(os.path.join(self.root_dir,"Masks"))
      self.mask_path = os.path.join(self.root_dir,"Masks")
    self.cow_teat_binary = cow_teat_binary
  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    img_path= os.path.join(self.root_dir,"Images",self.image_list[idx])
    mask_path=os.path.join(self.mask_path ,self.image_list[idx])

    # image = np.asarray(Image.open(img_path).convert ("RGB"))
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = np.asarray(Image.open(mask_path).convert('L'))
    mask = cv2.imread(mask_path, 0)
    if self.cow_teat_binary:
      mask[mask<61]=0
      mask[mask>=61]=1
      
    if self. transform:
      transformed = self.transform(image=image, mask=mask)
      image = transformed['image']
      mask = transformed['mask']

    return image, mask


# Define the dataset and data loader
