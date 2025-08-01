import cv2
img = cv2.imread('/media/mspx/Elements1/mp3d/v1/scans/1pXnuDYAj8r/1pXnuDYAj8r/matterport_depth_images/0d3a3b42009441e2a02200423049d804_d0_0.png')
print(img[:,:,2].mean())