#initialize PCA with first 20 principal components

# Importing required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
img = cv2.imread('50+ HD Backgrounds Pack By Deepak Creations (1).jpg') #you can use any image you want.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
red,green,blue = cv2.split(img) 
rangeofpca = min(img.shape[0], img.shape[1])
for i in range(rangeofpca,rangeofpca+1,1):

    
    pca = PCA(i)

    #Applying to red channel and then applying inverse transform to transformed array.
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)

    #Applying to Green channel and then applying inverse transform to transformed array.
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)

    #Applying to Blue channel and then applying inverse transform to transformed array.
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)


    variance = pca.explained_variance_ratio_ 
    img_compressed = (np.dstack((red_inverted, green_inverted, blue_inverted))).astype(np.uint8)
    print(i)
    print(sum(variance))
    if sum(variance) >0.999:
        break
from PIL import Image
im1 = Image.fromarray(img_compressed)
im1.save(f"PCAcompress1899.jpg",optimize = True)
1188
0.9966090061915055
1425
0.9985776694767785
1662
0.9994897121642233
1899
0.999860557746942
2136
0.9999783503279406
