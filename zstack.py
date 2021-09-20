import cv2
import numpy as np
import time

import matplotlib.pyplot as plt

from fast_slic import Slic
from PIL import Image
'''
HDR but for focal positions
Goal is to produce an image where everything is sharp and in focus
Stack multiple images
Use the pixel from the sharpest image


'''

'''
Work to be done with the blending

Things don't line up perfectly, I think this is because the lens dicatates where light goes
Can we try to correct for the lens at the beginning so we start with images that line up?

Can also try using the dual pixel data, low disparity means in focus/sharp and should be included in the final image

Can we break up boxes with weak graidients?

semantic segmentation based on edges?

merge based on ratio of sobel 1 vs soble 2?

Try to warp one image to match the other? Close enough to approximate with homogrpahy?
Can we just crop to find the best match?
Do tone mapping to create a global color space?

For super pixel, check which canny lines they are surroudned by, match the majority so superpixel regions within the same bounds are matched
'''


PATH1 = '1.jpg'#'1.jpg'
PATH2 = '2.jpg'#'2.jpg'
SAVE_PATH = 'reverse_warp_combine.jpg'#'golf_combine.jpg'#'slic30_average_compact.jpg'#'neighbors_zstack_pullmap_100x100_thresh.png'
PULL_PATH = 'pullMap_thresh_100x100.png'
def main():

    image1 = cv2.imread(PATH1)
    image2 = cv2.imread(PATH2)
    rows, cols, channels = image1.shape
    #showImage(image1-image2,'sub')
    #average = (image1 + image2)/510
    #average = cv2.resize(average, (int(cols/4),int(rows/4) ))
    #showImage(average, 'average')
    #pullMap = cv2.imread(PULL_PATH,cv2.IMREAD_GRAYSCALE)
    #print(pullMap.shape)
    #showImage(pullMap*255)
    #zstack = combine(image1, image2,pullMap)
    
    M = crop(image1, image2)
    Minv = np.linalg.inv(M)
    dim = (image1.shape[1],image1.shape[0])
    image3 = cv2.warpPerspective(image1, M, dim)
    
    #showImage(image1)
    #showImage(image2)
    #showImage(image3, 'warped')
    #print(image3)
    #print(image3.shape)
    #diff1 = image1 - image3
    
    diff2 = image2 - image3
    
    #print(np.amin(diff2))
    #showImage(diff1, 'sub1')
    #showImage(diff2, 'sub2')
    
    #zstack = combine(image1, image2)
    # image1 gets warped to image2 (wider fov) 
    zstack = combine(image3, image2)
    #showImage(zstack)
    #blurZstack = cv2.GaussianBlur(zstack, (7,7),0)
    reverseProj = cv2.warpPerspective(zstack, Minv, dim)
    showImage(reverseProj/255)
    #cv2.imwrite(SAVE_PATH, zstack)
    cv2.imwrite(SAVE_PATH, reverseProj)
def combine(image1, image2, pullMap = None):
        grad1 = getEdges(image1,method='sobel')
        grad2 = getEdges(image2,method='sobel')
        #showImage(grad1)
        print('showImage done')
	#showImage(grad1)
	#showImage(grad2)
        out = np.zeros(image1.shape)
        rows, cols, channels = image1.shape

	
	#showImage(grid1Small, 'smal')
	#cv2.imwrite('edges1.png',grid1Small)
	#cv2.imwrite('edges2.png',grid2Small)

	#showImage(grid1Small/grid2Small, 'ratio')
	#ret, thresh1 = cv2.threshold(grad1, 100, 255, cv2.THRESH_BINARY)
	#ret, thresh2 = cv2.threshold(grad2, 100, 255, cv2.THRESH_BINARY)
	#showImage(grad1,'grad1')
	#showImage(thresh1, 'thresh1')
	#showImage(grad2,'grad2')
	#showImage(thresh2, 'thresh2')
        clusters = 50
        compactness=10#0.01
        ''' 
        for i in range(5):  
            assignment1 = getAssignment(image1, clusters, compactness)
            compactness*=10
            showImage((assignment1*7)/255, title=str(compactness))
        '''
        #checkAssign = assignment1
        #showImage(checkAssign/255)
        assignment1 = getAssignment(image1, clusters, compactness)
        assignment2 = getAssignment(image2, clusters, compactness)
        cv2.imwrite('assignment1.jpg', assignment1)
        cv2.imwrite('assignment2.jpg', assignment2)
        #showImage(assignment1/255,resize=False)
        #if pullMap is None:
        # default to second image, 
        pullMap1 = np.zeros(image1.shape[:2])

        pullMap2 = np.zeros(image2.shape[:2])
        #pullMap = convolveCombine(thresh1, thresh2, kernel=(100,100))
        #cv2.imwrite("pullMap_100x100.png", pullMap)
        
        # use the slic result to generate a pullmap
        # assignments are likely not the same, how to rememdy?
        for cluster in range(clusters):
            # use the clusters as a mask
            # compare the masked region in grad1 with grad1
            # set the masked version in out
            #clusterMask = np.where(assignment1==cluster, 1, 0)
            #edgeMask1 = np.where(clusterMask==1, grad1, 
            
            #edgeMask1 = clusterMask and grad1
            #edgeMask2 = clusterMask and grad2
            edgeMask11 = np.where(assignment1==cluster, grad1, 0)
            edgeMask12 = np.where(assignment1==cluster, grad2, 0)

            edgeMask21 = np.where(assignment2==cluster, grad1, 0)
            edgeMask22 = np.where(assignment2==cluster, grad2, 0)
            if np.sum(edgeMask11) > np.sum(edgeMask12):
                # pull from the first image
                pullMap1 += np.where(assignment1==cluster, 1, 0)
                #pullMap2 += np.where(assignment2==cluster, 1, 0)
            if np.sum(edgeMask21) > np.sum(edgeMask22):
                pullMap2 += np.where(assignment2==cluster, 1, 0)
            # maybe do this for both assignments then blend the image?
            
	
        #pullMap = processPullMap(pullMap)
        #showImage(pullMap1*255)
        cv2.imwrite('pullmap1.jpg', pullMap1*255)
        #showImage(pullMap2*255)
        cv2.imwrite('pullmap2.jpg', pullMap2*255)

	# blurSobel[row][col] = amount from image 1 vs image 2
	# 1 means pull from image 1, 0 means pull from image 2
	#createMap(grad1, grad2)
	# rowGrid = 32
	# colGrid = 32
	# rowStride = int(rows/rowGrid)
	# colStride = int(cols/colGrid) 
	# for row in range(rowGrid):
	# 	for col in range(colGrid):
	# 		average1 = np.sum(grad1[rowStride * row: rowStride * (row+1), col*colStride:(col+1) * colStride])
	# 		average2 = np.sum(grad2[rowStride * row: rowStride * (row+1), col*colStride:(col+1) * colStride])
	# 		if average1 > average2:
	# 			# use the first image
	# 			out[row*rowStride:(row+1)*rowStride, col*colStride:(col+1)*colStride] = image1[row*rowStride:(row+1)*rowStride, col*colStride:(col+1)*colStride]
	# 		else:
	# 			out[row*rowStride:(row+1)*rowStride, col*colStride:(col+1)*colStride] = image2[row*rowStride:(row+1)*rowStride, col*colStride:(col+1)*colStride]
        out1 = np.zeros(image1.shape)
        out2 = np.zeros(image1.shape)
        for i in range(3):
            # I want neighboring pixels to be from the same image
            # try blockwise instead of pixelwise
            out1[:,:,i] = np.where(pullMap1>0, image1[:,:,i], image2[:,:,i])
            out2[:,:,i] = np.where(pullMap2>0, image1[:,:,i], image2[:,:,i])

            #out = np.where(pullMap==1, image1, image2)
        out = (out1 + out2) / 2
        return out

def getEdges(image, method=None):
    # can use different methods here?
    if method is None or method == 'sobel':
        # calcualte the sobel for each image
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        blur1 = cv2.GaussianBlur(image, (3,3), 0)
        #blur2 = cv2.GaussianBlur(image2, (3,3), 0)

        gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
        #gray2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)

        grad_x1 = cv2.Sobel(gray1, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        grad_y1 = cv2.Sobel(gray1, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


        abs_grad_x1 = cv2.convertScaleAbs(grad_x1)
        abs_grad_y1 = cv2.convertScaleAbs(grad_y1)
        grad1 = cv2.addWeighted(abs_grad_x1, 0.5, abs_grad_y1, 0.5, 0)

        #grad_x2 = cv2.Sobel(gray2, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        #grad_y2 = cv2.Sobel(gray2, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


        #abs_grad_x2 = cv2.convertScaleAbs(grad_x2)
        #abs_grad_y2 = cv2.convertScaleAbs(grad_y2)
        #grad2 = cv2.addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0)
        return grad1
    elif method == 'canny':
        edges = cv2.Canny(image, 100, 200)
        return edges
'''
Create a map that tells the final image where to get each pixel for the final image
Sobel returns a very grainy image, try some smoothign techniques to maintain a pixelwise map

closing (dilation then erosion)

partition image into two categories
pull from image 1
pull form image 2
sharp lines should definitely be pulled from a certain image
check for distance to an anchor point?

check local neighborhood, sample from image 1 if there are stronger edges from 1 in neighborhood, else sammple from 2
'''
def createMap(sobel1, sobel2):
	rows, cols = sobel1.shape
	kernel = np.ones((5,5),np.uint8)
	#sobel1 += 100
	smallSobel1 = cv2.resize(sobel1, (int(cols/4),int(rows/4)))
	showImage(smallSobel1)
	ret, thresh1 = cv2.threshold(sobel1, 100, 255, cv2.THRESH_BINARY)
	ret, thresh2 = cv2.threshold(sobel2, 100, 255, cv2.THRESH_BINARY)
	
	smallThresh = cv2.resize(thresh1, (int(cols/4),int(rows/4)))
	showImage(smallThresh, 'threshold1')
	dilation = cv2.dilate(thresh1,kernel,iterations = 1)
	smallDilate = cv2.resize(dilation, (int(cols/4),int(rows/4)))
	showImage(smallDilate, 'dilate')


	smallThresh = cv2.resize(thresh2, (int(cols/4),int(rows/4)))
	showImage(smallThresh, 'threshold2')
	dilation = cv2.dilate(thresh2,kernel,iterations = 1)
	smallDilate = cv2.resize(dilation, (int(cols/4),int(rows/4)))
	showImage(smallDilate, 'dilate')

	ratio = np.subtract(thre, sobel2)
	#print(ratio.shape)
	#print(ratio)
	# 255/2 ~ 128
	#ratio = np.where(ratio>128, 255, 0)
	ret, thresh = cv2.threshold(ratio, 127, 255, cv2.THRESH_BINARY)
	
	smallThresh = cv2.resize(thresh, (int(cols/4),int(rows/4)))
	showImage(smallThresh, 'threshold')
	#print(np.amin(ratio))
	
	#blurSobel = cv2.GaussianBlur(ratio, (5,5),0)
	# dilation, try to expand edges
	

	dilation = cv2.dilate(ratio,kernel,iterations = 1)
	smallDilate = cv2.resize(dilation, (int(cols/4),int(rows/4)))
	showImage(smallDilate, 'dilate')
def convolveCombine(sobel1, sobel2,kernel=(20,20)):
	rows, cols = sobel1.shape
	out = np.zeros(sobel1.shape)
	kRows,kCols = kernel
	leftRow = kRows//2
	rightRow = kRows-leftRow
	upCol = kCols//2
	downCol = kCols-upCol
	start = time.time()
	for row in np.arange(0,rows):
		startRow = max(0,row-leftRow)
		endRow = min(rows-1, row+rightRow)
		for col in np.arange(0,cols):
			startCol = max(0,col-upCol)
			endCol = min(cols-1, col+downCol)
			firstNeighbors = np.sum(sobel1[startRow:endRow, startCol:endCol])
			secondNeighbors = np.sum(sobel2[startRow:endRow, startCol:endCol])
			out[row][col] = 1 if firstNeighbors > secondNeighbors else 0
	print('finished manual convolution in ',(time.time() - start), 'seconds')
	showImage(out)
	return out
'''
Use SLIC to get a super pixel assignment, maybe can incorporate sobel info later?

'''
def getAssignment(image,clusters=100,compactness=10):
    convertColor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #pilImage = Image.fromarray(convertColor)
    slic = Slic(num_components=clusters, compactness=compactness)
    assignment = slic.iterate(convertColor)
    return assignment
def processPullMap(pullMap):
	showImage(pullMap*255,'before')
	kernelSize = (50,50)
	kernel = np.ones(kernelSize, np.uint8)
	closing = cv2.morphologyEx(pullMap*255, cv2.MORPH_CLOSE, kernel)
	showImage(closing, 'after')
	return closing

def showImage(image, title='image', resize=True):
    imageShape = image.shape
    rows = imageShape[0]
    cols = imageShape[1]
    if resize:
        newRows = int(rows/4)
        newCols = int(cols/4)
        showImage = cv2.resize(image,(newCols, newRows))
    else:
        showImage = image
    cv2.imshow(title, showImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
Assume image1 is with closer objects in focus and image2 is for farther objects
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
'''
def crop(image1, image2):
    # we should know which one is closer
    # farther objects need a closer lens
    # 1/Z + 1/r = 1/f
    # try to crop or warp the farther image to the closer
    
    # same lightfield, there shouldn't be any information that one view can see that the other cannot
    # same points map to different pixels
    
    # do we want to downsample the larger to the smaller?
    # find homography that maps 
    
    # use smaller image as a template, crop bigger one based on this
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #print(M)
        #print(mask)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        #print(dst)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    #plt.imshow(img3, 'gray'),plt.show()
    return M#cv2.warpPerspective(img1, M, (img1.shape[1],img1.shape[0])) 
if __name__=='__main__':
	main()
