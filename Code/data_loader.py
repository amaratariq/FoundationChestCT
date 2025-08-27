from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import nibabel as nib
import sys

class CTDataset(Dataset):
    def __init__(self, df, volumes_per_batch, batch_size_int, slice_gap, mask_fraction, image_size, n_channels, random_vol):
        
        self.df = df
        self.volumes_per_batch = volumes_per_batch
        self.batch_size = batch_size_int
        self.slice_gap = slice_gap
        self.mask_fraction = mask_fraction
        self.image_size = image_size
        self.n_channels = n_channels
        self.randomize_vol_selection = random_vol
        if self.mask_fraction is not None:
            self.mask_size = int((self.mask_fraction/100)*self.image_size)
        else:
            self.mask_size = None
        self.margin=50
        
    def __len__(self):
        return len(self.df)-self.volumes_per_batch  
    
    def vol_window(self, vol, level=50, window=350):
        maxval = level + window/2
        minval = level - window/2
        vol[vol<minval] = minval
        vol[vol>maxval] = maxval
        return vol

    def image_loader(self, image1, image2):
        trans = transforms.Compose([
                                    transforms.Resize((self.image_size, self.image_size)),
                                    transforms.CenterCrop(self.image_size),
                                    transforms.ToTensor(),
                                    ])

        a = image1.astype(float)
        a = (a-np.min(a))/(np.max(a)-np.min(a))
        a = 255.0*a
        r = None
        c = None

        if self.mask_size is not None:
            r = np.random.randint(self.image_size-self.mask_size-(2*self.margin))
            c = np.random.randint(self.image_size-self.mask_size-(2*self.margin))
            a[r:r+self.mask_size, c:c+self.mask_size] = 0
        if self.n_channels==3:
            image1 = Image.fromarray(np.uint8(a)).convert('RGB')
        else:
            image1 = Image.fromarray(np.uint8(a))        
        
        a = image2.astype(float)
        a = (a-np.min(a))/(np.max(a)-np.min(a))       
        a = 255.0*a
        if self.n_channels==3:
            image2 = Image.fromarray(np.uint8(a)).convert('RGB')
        else:
            image2 = Image.fromarray(np.uint8(a))                
        
        
        image1 = trans(image1).float() 
        image2 = trans(image2).float()   
        

        return image1, image2, r, c 
    
    def __getitem__(self, index):
        volumes = {}
        volumes_length = {}
        k = 0
        idx = (index*self.volumes_per_batch)%len(self.df)
        if self.randomize_vol_selection==False:
            while len(volumes_length) < self.volumes_per_batch:
                try:
                    vol = nib.load(self.df.at[self.df.index[idx], 'volume_location'])
                    vol = np.array(vol.dataobj)
                    vol = self.vol_window(vol)
                    vol = np.rot90(vol, k =1)  ##### spine at the image base
                    volumes[k] = vol.copy()
                    volumes_length[k] = vol.shape[2]
                    k+=1
                    idx+=1
                except:
                    print(self.df.index[idx],  self.df.at[self.df.index[idx], 'volume_location'], 'PROBLEM')
                    idx+=1
                idx = idx%len(self.df)
        else:
            while len(volumes_length) < self.volumes_per_batch:
                try:
                    rand_idx = np.random.choice(len(self.df), 1)[0]
                    vol = nib.load(self.df.at[self.df.index[rand_idx], 'volume_location'])
                    vol = np.array(vol.dataobj)
                    vol = self.vol_window(vol)
                    vol = np.rot90(vol, k =1) ##### spine at the image base
                    volumes[k] = vol.copy()
                    volumes_length[k] = vol.shape[2]
                    k+=1
                    #idx+=1
                except:
                    print(self.df.index[rand_idx], self.df.at[self.df.index[rand_idx], 'volume_location'], 'PROBLEM')
                    #idx+=1
                #idx = idx%len(self.df)
        X = None
        Y = None
        R = []
        C = []
        i = 0
        while i < self.batch_size:
            try:
                v_range = len(volumes)
                vid = np.random.choice(v_range, 1)[0]
                s_range = volumes_length[vid]
                sid = np.random.choice(int(s_range-self.slice_gap), 1)[0]

                x = volumes[vid][:,:,sid]
                y = volumes[vid][:,:,sid+self.slice_gap]

                x, y, r, c = self.image_loader(x, y) 

                if X == None:
                    X = torch.clone(x)
                    Y = torch.clone(y)
                    X = X[None, :,:]
                    Y = Y[None, :,:]
                    R.append(r)
                    C.append(c)
                else:
                    X = torch.cat((X, torch.clone(x[None, :,:])), 0)
                    Y = torch.cat((Y, torch.clone(y[None, :,:])), 0)
                    R.append(r)
                    C.append(c)
                i+=1
            except:
                print(v_range, vid, s_range, self.slice_gap, 'PROBLEM')
            sys.stdout.flush()
        return X, Y, torch.Tensor(R), torch.Tensor(C)