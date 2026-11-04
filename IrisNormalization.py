"""Rubber-sheet normalization: 64 radial x 512 angular samples (Ma 2003)."""
import cv2
import numpy as np

def IrisNormalization(img,pupil,iris,angle=0.0,shape=(64,512)):
    H,W=shape
    theta=np.arange(W,dtype=np.float32)*(2*np.pi/W)+np.deg2rad(angle)
    t=(np.arange(H,dtype=np.float32)+0.5)/H
    px,py,pr=pupil; ix,iy,ir=iris
    xx=(px+pr*np.cos(theta))[None,:]*(1-t[:,None])+(ix+ir*np.cos(theta))[None,:]*t[:,None]
    yy=(py+pr*np.sin(theta))[None,:]*(1-t[:,None])+(iy+ir*np.sin(theta))[None,:]*t[:,None]
    return cv2.remap(img,xx.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT).astype(np.uint8)
