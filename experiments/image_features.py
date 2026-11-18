"""Exact eye primitives and original team global descriptor."""
import numpy as np
from PIL import Image,ImageOps
from scipy.ndimage import map_coordinates
from skimage.feature import local_binary_pattern
from owner_dog.features import extract_variants

def align_eye(im,points):
    im=ImageOps.exif_transpose(im).convert('RGBA');w,h=im.size
    p=np.asarray(points,dtype=float)
    if p.shape!=(4,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Eye coordinates must be four finite normalized coordinates in [0,1]')
    pts=p.reshape(2,2)*[w,h]
    if pts[1,0]<=pts[0,0] or np.linalg.norm(pts[1]-pts[0])<4:
        raise ValueError('Eye centers must be image-left then image-right with >=4 pixels separation')
    dx,dy=(pts[1]-pts[0])/48
    a,b=dx,-dy;d,e=dy,dx
    c=pts[0,0]-a*24-b*16;f=pts[0,1]-d*24-e*16
    crop=im.transform((96,32),Image.Transform.AFFINE,(a,b,c,d,e,f),Image.Resampling.BICUBIC,fillcolor=(255,255,255,255))
    rgba=np.asarray(crop,dtype=float)/255
    return rgba[:,:,:3]*rgba[:,:,3,None]+1-rgba[:,:,3,None]

def eye_patches(im,points):
    rgb=align_eye(im,np.asarray(points)/128)
    p=np.asarray(points).reshape(2,2)*2;dx,dy=(p[1]-p[0])/48
    a,b=dx,-dy;d,e=dy,dx;c=p[0,0]-a*24-b*16;f=p[0,1]-d*24-e*16
    alpha=np.asarray(im.getchannel('A').transform((96,32),Image.Transform.AFFINE,(a,b,c,d,e,f),Image.Resampling.BICUBIC,fillcolor=0))/255
    grays=[];alphas=[]
    for xs in [slice(8,40),slice(56,88)]:
        grays.append(np.round((rgb[4:28,xs]@[.299,.587,.114])*255).astype('uint8'))
        alphas.append(alpha[4:28,xs])
    return np.stack(grays),np.stack(alphas)

def neighbors(gray):
    yy,xx=np.indices(gray.shape);n=[]
    for p in range(8):
        theta=2*np.pi*p/8
        n.append(map_coordinates(gray.astype(float),[yy-np.sin(theta),xx+np.cos(theta)],order=1,mode='reflect'))
    return np.stack(n)

def hist80(grays,kind='baseline'):
    parts=[]
    yy,xx=np.indices((24,32));core=((xx-15.5)/9)**2+((yy-11.5)/5.5)**2<=1
    for gray in grays:
        code=local_binary_pattern(gray,8,1,'uniform');n=neighbors(gray);mag=np.abs(n-gray).mean(axis=0)
        if kind=='ltp':
            # Tan/Triggs-inspired split ternary pattern, project implementation.
            # Fixed +/-5 gray-level dead zone; sign and uniform mapping applied separately.
            sets=[]
            for bits in [n>=gray+5.,n<=gray-5.]:
                flips=(bits!=np.roll(bits,1,axis=0)).sum(axis=0);ones=bits.sum(axis=0)
                sets.append(np.where(flips<=2,ones,9))
        else:sets=[code]
        for codes in sets:
            for y in range(2):
                for x in range(2):
                    mask=np.zeros((24,32),bool);mask[y*12:(y+1)*12,x*16:(x+1)*16]=True
                    if kind=='core':mask&=core
                    if kind=='ring':mask&=~core
                    weights=mag[mask] if kind=='contrast_weighted' else None
                    h=np.bincount(codes[mask].astype(int),weights=weights,minlength=10).astype(float)
                    if h.sum()<=1e-12:h[8]=1
                    parts.extend(np.sqrt(h/h.sum()))
    p=np.array(parts);return p/max(np.linalg.norm(p),1e-12)
