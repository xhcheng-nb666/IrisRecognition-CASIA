"""Ma 2003: 16x16 illumination estimation; local 32x32 histogram equalization."""
import cv2
import numpy as np

def ImageEnhancement(norm):
    arr=norm.astype(np.float32)
    background=cv2.resize(cv2.resize(arr,(32,4),interpolation=cv2.INTER_AREA), (512,64),interpolation=cv2.INTER_CUBIC)
    corrected=arr-background
    corrected=np.clip(corrected+128,0,255).astype(np.uint8)
    out=np.empty_like(corrected)
    for r in range(0,64,32):
        for c in range(0,512,32):
            out[r:r+32,c:c+32]=cv2.equalizeHist(corrected[r:r+32,c:c+32])
    return out
