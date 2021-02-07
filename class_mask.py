import cv2
import numpy as np

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v


def final_masked_image(image,mask_image,m_rgb,rgb):
    
    #Make an array of image with zeros
    image_zero=image.copy()*0
    
    # choose color of the mask bgr and replace it with 255 or 0
    final_mask=np.where(mask_image==m_rgb,255,image_zero)
    
    # Bitwise and to place mask over image
    masked_image= cv2.bitwise_and(image, final_mask)

    
    #Find height width and channel of the image
    height,width,ch = image.shape
    
    #Change the colour of image to HSV from BGR
    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_image)
    
    #Get the resultant HSV values from RGB
    aa=rgb_to_hsv(rgb[0],rgb[1],rgb[2])
        
    #HUE adjustments
    hsv_image[:,:,0] = (np.ones(shape=(height,width))[:,:])*int(aa[0]//2) # Changes the H value
    
    
    #SATURATION adjustments
    # hsv_image[:,:,1] = np.ones(shape=(height,width))*int(aa[1]) # Changes the S value
  
    
    #VALUE adjustments
    value_new = (max(rgb)//255)*100    
    # hsv_image[:,:] = np.where(mask_image==m_rgb,value_new,hsv_image[:,:])
    
    #Change the color of image to BGR from HSV
    bgr_image= cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR) 
    
    #combine modified and orignal image
    final_image=cv2.addWeighted(image,1,bgr_image,1,0)
    
    return final_image,value_new
    
    
image = cv2.imread('org_1.jpg')
mask_image= cv2.imread('mask_1.png')

result,kk=final_masked_image(image,mask_image,(210,225,27),(255,0,0))
result_1,jj=final_masked_image(result,mask_image,(75,194,108),(0,255,0))

cv2.imshow("images", np.hstack([image,mask_image,result_1]))
cv2.imwrite("images.png", np.hstack([image,mask_image,result,result_1]))
cv2.waitKey(0)
cv2.destroyAllWindows()
    #BGR
    # 90,6,69 purple bg
    #27,225, 210 coat yellow
    #108 ,194,75skirt green
    #rgb