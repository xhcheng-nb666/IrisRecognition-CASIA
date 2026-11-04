"""Pupil and limbus localization for CASIA-IrisV1 (circles: x,y,r)."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def IrisLocalization(img):
    h, w = img.shape
    blur = cv2.GaussianBlur(img, (5,5), 1.1)
    # CASIA V1 pupils have been replaced with a nearly uniform dark disk.
    _, mask = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        cx,cy=centers[i]
        if 400 < area < 16000 and 0.62 < bw/max(bh,1) < 1.55 and 25 < cx < w-25 and 25 < cy < h-25:
            circularity=area/(np.pi*(max(bw,bh)/2)**2)
            candidates.append((area*min(circularity,1.0),cx,cy,np.sqrt(area/np.pi)))
    if candidates:
        _,px,py,pr=max(candidates)
    else:
        # fallback: circular Hough detection
        c=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1.2,minDist=60,param1=80,param2=28,minRadius=18,maxRadius=60)
        if c is None:
            raise ValueError('Pupil detection failed')
        px,py,pr=min(c[0],key=lambda z:abs(z[0]-w/2)+abs(z[1]-h/2))
    # Gradient-integral circle search (outer boundary), favor left/right iris arcs
    grad_x=cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)
    grad_y=cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
    mag=cv2.magnitude(grad_x,grad_y)
    best=(-1,None)
    angles=np.r_[np.linspace(-0.58,0.58,90),np.linspace(np.pi-0.58,np.pi+0.58,90)]
    for dx in range(-5,6,2):
        for dy in range(-5,6,2):
            cx,cy=px+dx,py+dy
            radii=np.arange(max(int(pr+27),65),min(int(pr+91),116))
            if len(radii)==0:continue
            xs=cx+radii[:,None]*np.cos(angles)[None,:]
            ys=cy+radii[:,None]*np.sin(angles)[None,:]
            valid=(xs>3)&(xs<w-4)&(ys>3)&(ys<h-4)
            samples=cv2.remap(mag,xs.astype(np.float32),ys.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            scores=np.sum(samples*valid,axis=1)/np.maximum(valid.sum(axis=1),1)
            j=np.argmax(gaussian_filter1d(scores,1))
            score=scores[j]
            if score>best[0]:best=(score,(cx,cy,float(radii[j])))
    if best[1] is None:raise ValueError('Outer iris boundary detection failed')
    return (float(px),float(py),float(pr)),best[1]
