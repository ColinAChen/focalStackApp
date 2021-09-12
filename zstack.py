import cv2
import numpy as np
import time
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
'''
PATH1 = '1.jpg'
PATH2 = '2.jpg'
SAVE_PATH = 'neighbors_zstack_pullmap_100x100_thresh.png'
PULL_PATH = 'pullMap_thresh_100x100.png'
def main():

	image1 = cv2.imread(PATH1)
	image2 = cv2.imread(PATH2)
	rows, cols, channels = image1.shape
	average = (image1 + image2)/510
	average = cv2.resize(average, (int(cols/4),int(rows/4) ))
	#showImage(average, 'average')
	pullMap = cv2.imread(PULL_PATH,cv2.IMREAD_GRAYSCALE)
	#print(pullMap.shape)
	#showImage(pullMap*255)
	#zstack = combine(image1, image2,pullMap)
	zstack = combine(image1, image2)
	#showImage(zstack)
	#blurZstack = cv2.GaussianBlur(zstack, (7,7),0)

	cv2.imwrite(SAVE_PATH, zstack)
def combine(image1, image2, pullMap = None):
	# calcualte the sobel for each image
	scale = 1
	delta = 0
	ddepth = cv2.CV_16S

	blur1 = cv2.GaussianBlur(image1, (3,3), 0)
	blur2 = cv2.GaussianBlur(image2, (3,3), 0)

	gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)

	grad_x1 = cv2.Sobel(gray1, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
	# Gradient-Y
	# grad_y = cv.Scharr(gray,ddepth,0,1)
	grad_y1 = cv2.Sobel(gray1, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


	abs_grad_x1 = cv2.convertScaleAbs(grad_x1)
	abs_grad_y1 = cv2.convertScaleAbs(grad_y1)
	grad1 = cv2.addWeighted(abs_grad_x1, 0.5, abs_grad_y1, 0.5, 0)

	grad_x2 = cv2.Sobel(gray2, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
	# Gradient-Y
	# grad_y = cv.Scharr(gray,ddepth,0,1)
	grad_y2 = cv2.Sobel(gray2, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


	abs_grad_x2 = cv2.convertScaleAbs(grad_x2)
	abs_grad_y2 = cv2.convertScaleAbs(grad_y2)
	grad2 = cv2.addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0)


	#showImage(grad1)
	#showImage(grad2)
	out = np.zeros(image1.shape)
	rows, cols, channels = image1.shape

	
	#showImage(grid1Small, 'smal')
	#cv2.imwrite('edges1.png',grid1Small)
	#cv2.imwrite('edges2.png',grid2Small)

	#showImage(grid1Small/grid2Small, 'ratio')
	ret, thresh1 = cv2.threshold(grad1, 100, 255, cv2.THRESH_BINARY)
	ret, thresh2 = cv2.threshold(grad2, 100, 255, cv2.THRESH_BINARY)
	showImage(grad1,'grad1')
	showImage(thresh1, 'thresh1')
	showImage(grad2,'grad2')
	showImage(thresh2, 'thresh2')
	if pullMap is None:
		pullMap = convolveCombine(thresh1, thresh2, kernel=(100,100))
		cv2.imwrite("pullMap_100x100.png", pullMap)
	
	pullMap = processPullMap(pullMap)
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
	for i in range(3):
		# I want neighboring pixels to be from the same image
		# try blockwise instead of pixelwise
		out[:,:,i] = np.where(pullMap>0, image1[:,:,i], image2[:,:,i])
		#out = np.where(pullMap==1, image1, image2)
	return out
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
if __name__=='__main__':
	main()