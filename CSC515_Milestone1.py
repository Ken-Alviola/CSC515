#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[3]:


img = cv2.imread('brain.jpg')
cv2.imwrite('brain_copy.png',img)
cv2.imshow('brain_window',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




