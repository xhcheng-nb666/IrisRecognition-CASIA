"""Circular-modulation Gaussian spatial filters, two channels, 8x8 statistics."""
import numpy as np
import cv2
from functools import lru_cache

@lru_cache(None)
def kernel(sx, sy, frequency):

    # Create coordinates for the spatial filter
    xx, yy = np.meshgrid(
        np.arange(-15, 16, dtype=np.float64),
        np.arange(-8, 9, dtype=np.float64)
    )

    # Gaussian envelope
    gaussian = (
        1 / (2 * np.pi * sx * sy)
    ) * np.exp(
        -0.5 * (
            (xx / sx) ** 2 +
            (yy / sy) ** 2
        )
    )

    # Circularly symmetric sinusoidal modulation
    modulation = np.cos(
        2 * np.pi * frequency *
        np.sqrt(xx ** 2 + yy ** 2)
    )

    # Spatial filter from Ma et al. (2003), Equation (3)
    g = gaussian * modulation

    return g.astype(np.float32)

def FeatureExtraction(enhanced,frequency=0.15):
    roi=enhanced[:48,:].astype(np.float32)
    vectors=[]
    for sx,sy in [(3.0,1.5),(4.5,1.5)]:
        filtered=np.abs(cv2.filter2D(roi,cv2.CV_32F,kernel(sx,sy,frequency),borderType=cv2.BORDER_REFLECT))
        blocks=filtered.reshape(6,8,64,8).transpose(0,2,1,3)
        mean=blocks.mean(axis=(2,3))
        mad=np.abs(blocks-mean[:,:,None,None]).mean(axis=(2,3))
        vectors.extend(np.stack([mean,mad],axis=-1).reshape(-1))
    return np.asarray(vectors,dtype=np.float32)
