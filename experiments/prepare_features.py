"""Rebuild or spot-check included features from original PNGs and eye labels."""
from pathlib import Path
import argparse,json,io
import numpy as np
from PIL import Image
from owner_dog.data import DatasetSource
from owner_dog import engine
from image_features import extract_variants,eye_patches,hist80
BASE=Path(__file__).resolve().parent
def extract(ds,key,annotations):
    pair,role,photo=key
    image=Image.open(io.BytesIO(ds.read(key))).convert('RGBA')
    full=extract_variants(image)['enhanced']
    points=annotations[str(pair)][photo-1+(0 if role=='person' else 3)]
    if points is None:eye=np.full(20,np.nan)
    else:
        gray,_=eye_patches(image,np.asarray(points));a=hist80(gray)
        eye=np.sqrt((a*a).reshape(2,4,10).sum(axis=1)).ravel()
    return full,eye
def main(check):
    ann=json.loads((BASE/'data/eye_annotations.json').read_text(encoding='utf8'))['points']
    old=dict(np.load(BASE/'data/features.npz'));made={k:np.full_like(v,np.nan) for k,v in old.items()};errors=[];count=0
    with DatasetSource(BASE/'data/학습용_200세트.zip',200) as ds:
        keys=[(i,r,1) for i in [1,18,52,100,150,200] for r in ['person','dog']]+[(70,'dog',3),(147,'dog',3)] if check else [(row['pair_id'],row['role'],row['photo_index']) for row in ds.records]
        for key in keys:
            full,eye=extract(ds,key,ann);i=key[0]-1;j=key[2]-1;role=key[1]
            assert np.allclose(full,old[role+'__global'][i,j],atol=1e-12,rtol=0)
            assert np.allclose(eye,old[role+'__eye'][i,j],atol=1e-12,rtol=0,equal_nan=True)
            errors.append(float(np.max(np.abs(full-old[role+'__global'][i,j]))))
            made[role+'__global'][i,j]=full;made[role+'__eye'][i,j]=eye;count+=1
            if count%100==0:print('images',count,flush=True)
    result=dict(images_checked=count,max_global_absolute_error=max(errors),invisible_eye_examples_checked=True,passed=True)
    out=BASE/'outputs';out.mkdir(exist_ok=True)
    if not check:np.savez_compressed(out/'rebuilt_features.npz',**made)
    (out/'feature_check.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-samples',action='store_true');args=p.parse_args();main(args.check_samples)
