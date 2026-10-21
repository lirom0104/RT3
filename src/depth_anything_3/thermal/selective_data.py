"""Original VGGT replay plus one scene with solid depth and sky ordering labels."""
import json,math
from pathlib import Path
import numpy as np
from PIL import ImageEnhance
import torch
from .hybrid_data import SEARDataset,read_frame,read_supervision_weight

class SelectiveDataset(SEARDataset):
    def __init__(self,*args,repair_split=None,**kwargs):
        super().__init__(*args,**kwargs)
        if repair_split:
            self.frames=[[f for f in frames if f.get('repair_split')==repair_split] for frames in self.frames]
            if any(len(f)<self.views for f in self.frames):raise ValueError('Insufficient split frames')
        self.items=[]
        for i,frames in enumerate(self.frames):
            self.items.extend((i,j) for j in range(math.ceil(len(frames)/self.views) if self.training else 3))

    def __getitem__(self,index):
        si,ji=self.items[index];frames=self.frames[si];scene=self.scenes[si];folder=Path(scene['transforms']).parent
        h,w=frames[0]['rgb']['h'],frames[0]['rgb']['w']
        if h%14 or w%14:raise ValueError((h,w))
        rng=np.random.default_rng(self.seed+index*1009+self.epoch*1000003)
        if self.training:
            span=min(len(frames),self.views*4);start=int(rng.integers(len(frames)-span+1))
            ids=rng.choice(np.arange(start,start+span),self.views,replace=False)
            # Explicit equal exposure to RGB-only, mixed, and thermal-only contexts.
            mode=('rgb','mixed','thermal')[(index+self.epoch)%3]
        else:ids=np.linspace(0,len(frames)-1,self.views,dtype=int);mode=('rgb','mixed','thermal')[ji]
        mask=np.zeros(self.views,dtype=bool)
        if mode=='thermal':mask[:]=True
        if mode=='mixed':mask[::2]=True
        values={k:[] for k in ('images','depth','extrinsics','intrinsics','supervision_weight','soft_depth','sky_mask','sky_lower')}
        gain,gamma=rng.uniform(.95,1.05),rng.uniform(.95,1.05)
        for j,i in enumerate(ids):
            f=frames[int(i)]['thermal' if mask[j] else 'rgb']
            im,d,e,k=read_frame(folder,f,h,w);weight=read_supervision_weight(folder,f,h,w,d)
            soft=np.load(folder/f['soft_depth_file_path']).squeeze().astype('float32') if 'soft_depth_file_path'in f else d.copy()
            sky=np.load(folder/f['sky_mask_file_path']).astype(bool) if 'sky_mask_file_path'in f else np.zeros((h,w),bool)
            if self.training and not mask[j]:im=ImageEnhance.Brightness(im).enhance(gain)
            a=np.asarray(im).astype('float32')/255
            if self.training and mask[j]:a=np.clip(gain*np.power(a,gamma),0,1)
            for key,v in {'images':torch.from_numpy(a.copy()).permute(2,0,1),'depth':torch.from_numpy(d.copy()),'extrinsics':torch.from_numpy(e),'intrinsics':torch.from_numpy(k),'supervision_weight':torch.from_numpy(weight),'soft_depth':torch.from_numpy(soft),'sky_mask':torch.from_numpy(sky),'sky_lower':torch.tensor(f.get('sky_lower_bound',0),dtype=torch.float32)}.items():values[key].append(v)
        b={k:torch.stack(v) for k,v in values.items()}
        b['images']=(b['images']-b['images'].new_tensor([.485,.456,.406])[None,:,None,None])/b['images'].new_tensor([.229,.224,.225])[None,:,None,None]
        valid=torch.isfinite(b['depth'])&(b['depth']>1e-6)&(b['supervision_weight']>0)
        if valid.sum()<100:raise ValueError(scene['scene'])
        b['depth']=torch.where(valid,b['depth'],0);scale=b['depth'][valid].median()
        for k in ('depth','soft_depth','sky_lower'):b[k]/=scale
        b['extrinsics']=b['extrinsics']@torch.linalg.inv(b['extrinsics'][0]);b['extrinsics'][:,:3,3]/=scale
        b.update(valid=valid,hard_valid=valid&(b['supervision_weight']>=.99),thermal_mask=torch.from_numpy(mask),scene=scene['scene'],mode=mode)
        return b
