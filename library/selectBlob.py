#==============================================================================
#                              Select Blob Module
#==============================================================================
# Inputs:
# img_in			2D array of UINT8			binarized image (blobs)
# keypoints			array of keypoints			desired blobs keypoints
#------------------------------------------------------------------------------
# Outputs:
# img_out			2D array of UINT8			output image (selected blobs)
#==============================================================================

#------------------------------------------------------------------------------
# Load Dependencies
#------------------------------------------------------------------------------
import cv2
import numpy as np

#------------------------------------------------------------------------------
# Function selectBlob()
#------------------------------------------------------------------------------
def selectBlob(img_in, keypoints):

	# Label each independent blob with a number (labels)
	num_labels, labels = cv2.connectedComponents(img_in)

	# Allocatemq memory for result image
	img_out = np.zeros_like(img_in, dtype = np.uint8)

	# Test if any keypoint is given (only operates if so)
	if len(keypoints) > 0:

		# >>> Select blobs based on keypoint values <<<

		# Iterate through all keypoints and select desired labels
		for KP in keypoints:
			line = int(KP.pt[0])
			column = int(KP.pt[1])
			selected_label = labels[column, line]
			
			# Insert each selected blob in the result image
			img_label = np.where(labels == selected_label, 255, 0).astype('uint8')
			img_out = np.bitwise_or(img_out, img_label)

	# Return result binary image (selected blobs)
	return img_out
#------------------------------------------------------------------------------