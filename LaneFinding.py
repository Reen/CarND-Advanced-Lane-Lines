import numpy as np
from moviepy.editor import VideoFileClip
import pickle
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import inv
from glob import glob

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        # the Kalman Filter instance keeping track of the fit parameters
        self.filter = None
    
    def initialize(self, fit):
        self.filter = KalmanFilter(fit)
        self.detected = True
        self.update_curvature()
    
    def update(self, fit):
        self.filter.predict_and_update(fit)
        self.update_curvature()

    def invalidate(self):
        self.detected = False

    def update_curvature(self):
        y, x = self.to_pixels()

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(y*ImageProcessing.ym_per_pix, x*ImageProcessing.xm_per_pix, 2)
        # Calculate the new radii of curvature
        y_eval = np.max(y)
        curvature = ((1 + (2*left_fit_cr[0]*y_eval*ImageProcessing.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        # Now our radius of curvature is in meters
        self.radius_of_curvature = curvature
    
    def to_pixels(self, ploty=None):
        shape_x, shape_y = ImageProcessing.warped_image_shape
        if ploty is None:
            ploty = np.linspace(0, shape_y - 1, shape_y)
        plotyy = shape_y - ploty
        fit = self.filter.x
        fitx = shape_x - (fit[0]*plotyy**2 + fit[1]*plotyy + fit[2])
        return ploty, fitx
        

class KalmanFilter(object):
    def __init__(self, fit):
        self.x = fit
        self.P = np.array([
            [0.03, 0.0, 0.0],
            [0.0,   1, 0.0], 
            [0.0,  0.0,  10]]) # initial covariance
        self.F = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]])   # state transition matrix
        self.H = np.diag([1.0, 1.0, 1.0])                 # measurement function, in this case unit matrix as measurement space equals state space
        self.R = np.diag([0.02, 0.1, 4])               # measurement uncertainty
        self.Q = np.array([
            [0.001,  0, 0.0],
            [0,   0.1, 0.0],
            [0.0,  0.0, 1.5]])      # process noise

    # kalman filter taken and adapted from book "Kalman and Bayesian Filters in Python" by Roger R. Labbe 
    # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/
    def predict_and_update(self, z):
        # predict
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q
        
        #update
        S = np.dot(self.H, self.P).dot(self.H.T) + self.R
        K = np.dot(self.P, self.H.T).dot(inv(S))
        y = z - np.dot(self.H, self.x)
        self.x += np.dot(K, y)
        self.P = self.P - np.dot(K, self.H).dot(self.P)


class CameraCalibration():
    def __init__(self, calibration_file):
        dist_pickle = pickle.load(open(calibration_file, "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]


class ImageProcessing():

    source_image_shape = (1280, 720)
    warped_image_shape = (1280, 720)

    ym_per_pix = (3+12)/(660-480) # meters per pixel in y dimension
    xm_per_pix = 3.7/(946-328)    # meters per pixel in x dimension
    
    def __init__(self):
        self.calibration = CameraCalibration("calibration.p")
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        source_points = np.array([(608, 440), (203, 720), (1127,720), (676,440)], dtype=np.float32)
        warped_points = np.array([(320, 0), (320, 720), (960,720), (960,0)], dtype=np.float32)
        
        self.M = cv2.getPerspectiveTransform(source_points, warped_points)
        self.Minv = cv2.getPerspectiveTransform(warped_points, source_points)

    def undistort(self, image):
        return cv2.undistort(image, self.calibration.mtx, self.calibration.dist, None, self.calibration.mtx)

    def mask_sobel(self, img, sx_thresh=(20, 100)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        return sxbinary
    
    def mask_color(self, img, c_thresh=(170, 255)):
        binary = np.zeros_like(img)
        binary[(img >= c_thresh[0]) & (img <= c_thresh[1])] = 1
        return binary

    def mask_color_sobel(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        # Convert to HSV color space and separate the V channel
        h, l, s = self.image_to_hls_components(img)
        
        sxbinary = self.mask_sobel(l, sx_thresh)
        
        s_binary = self.mask_color(s, s_thresh)

        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
        return color_binary
    
    def image_to_hls_components(self, img):
        return cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HLS))
    
    def image_to_yuv_components(self, img):
        return cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))

    def local_hist_equalize(self, img):
        y,u,v = self.image_to_yuv_components(img)
        y = self.clahe.apply(y)
        return cv2.cvtColor(cv2.merge((y,u,v)), cv2.COLOR_YUV2RGB)
    
    def perspective_transform(self, img):
        return cv2.warpPerspective(img, self.M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)

    def inverse_perspective_transform(self, img):
        return cv2.warpPerspective(img, self.Minv, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)


class LaneFinding():
    def __init__(self):
        self.imageProcessing = ImageProcessing()
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin = 100
        self.window_width = 50 
        self.window_height = 80
        self._haveLanes = False
        
        self._leftLane = Line()
        self._rightLane = Line()

    def process_image(self, image):
        undistorted = self.imageProcessing.undistort(image)
        birdseye = self.imageProcessing.perspective_transform(undistorted)
        masked = self.imageProcessing.mask_color_sobel(birdseye)
        combined_mask = masked[:,:, 0] | masked[:,:,1] | masked[:,:,2]
        if not self._haveLanes:
            fit_vis, left_fit, right_fit = self.sliding_window_fit(combined_mask)
            self._haveLanes = True
            self._leftLane.initialize(left_fit)
            self._rightLane.initialize(right_fit)
        else:
            fit_vis, left_fit, right_fit = self.refit(combined_mask)
            
            self._leftLane.update(left_fit)
            self._rightLane.update(right_fit)
        
        #window_centroids = self.find_window_centroids(combined_mask)
        #centroid_vis = self.render_window_centroids(255 * combined_mask, window_centroids)
        left_curv, right_curv = self._leftLane.radius_of_curvature, self._rightLane.radius_of_curvature
        offset = self.calculate_offset()

        # visualization
        two_thirds = (854, 480)
        one_third = (426, 240)
        cm = cv2.resize(combined_mask, one_third)
        cm = 255 * np.dstack((cm,cm,cm))
        self.render_car_view(undistorted, combined_mask)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undistorted,
                    'Curvature: Left: {0:.1f}m Right {1:.1f}m | Offset: {2:.2f}m'.format(left_curv, right_curv, offset),
                    (10,50), font, 1, (255,255,255), 2)
        return undistorted
        top = np.hstack([
            cv2.resize(undistorted, two_thirds),
            np.vstack([
                cv2.resize(birdseye, one_third),
                cm])
            ])
        bottom = np.hstack([
            cv2.resize(fit_vis, one_third),
            #cv2.resize(centroid_vis, one_third),
            np.zeros((one_third[1], 1280 - 1 * one_third[0], 3))
        ])
        return np.vstack([top, bottom])
    
    def calculate_offset(self):
        center = (self._rightLane.filter.x[2] + self._leftLane.filter.x[2]) / 2
        offset = 1280/2.0 - center
        return - offset * ImageProcessing.xm_per_pix

    def render_car_view(self, undistorted, warped):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty, leftx = self._leftLane.to_pixels()
        _, rightx = self._rightLane.to_pixels(ploty=ploty)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.imageProcessing.inverse_perspective_transform(color_warp)
        # Combine the result with the original image
        cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0, undistorted)

    def sliding_window_fit(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Fit a second order polynomial to extracted pixels
        left_nonzero_x, left_nonzero_y, left_fit = self.extract_pixels_and_fit(nonzerox, nonzeroy, left_lane_inds)
        right_nonzero_x, right_nonzero_y, right_fit = self.extract_pixels_and_fit(nonzerox, nonzeroy, right_lane_inds)

        out_img[left_nonzero_y, left_nonzero_x] = [255,0,0]
        out_img[right_nonzero_y, right_nonzero_x] = [0,0,255]
        self.plot_fitted_lane_line(out_img, left_fit, right_fit)

        return (out_img, left_fit, right_fit)
    
    def extract_pixels_and_fit(self, nonzerox, nonzeroy, lane_inds):
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        xx = ImageProcessing.warped_image_shape[0] - x
        yy = ImageProcessing.warped_image_shape[1] - y
        fit = np.polyfit(yy, xx, 2)
        return x, y, fit
    
    def refit(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        _, left_mid = self._leftLane.to_pixels(nonzeroy)
        _, right_mid = self._rightLane.to_pixels(nonzeroy)
        left_lane_inds = ((nonzerox > (left_mid - self.margin)) & (nonzerox < (left_mid + self.margin))) 
        right_lane_inds = ((nonzerox > (right_mid - self.margin)) & (nonzerox < (right_mid + self.margin)))

        # Fit a second order polynomial to extracted pixels
        left_nonzero_x, left_nonzero_y, left_fit = self.extract_pixels_and_fit(nonzerox, nonzeroy, left_lane_inds)
        right_nonzero_x, right_nonzero_y, right_fit = self.extract_pixels_and_fit(nonzerox, nonzeroy, right_lane_inds)

        # Visualization
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img[left_nonzero_y, left_nonzero_x] = [255,0,0]
        out_img[right_nonzero_y, right_nonzero_x] = [0,0,255]
        self.plot_fitted_lane_line(out_img, left_fit, right_fit, True)

        return out_img, left_fit, right_fit
    
    def fit_to_pixels(self, left_fit, right_fit, ploty=None):
        shape_x, shape_y = self.imageProcessing.warped_image_shape
        if ploty is None:
            ploty = np.linspace(0, shape_y - 1, shape_y)
        plotyy = shape_y - ploty
        left_fitx = shape_x - (left_fit[0]*plotyy**2 + left_fit[1]*plotyy + left_fit[2])
        right_fitx = shape_x - (right_fit[0]*plotyy**2 + right_fit[1]*plotyy + right_fit[2])
        return ploty, left_fitx, right_fitx

    def plot_fitted_lane_line(self, image, left_fit, right_fit, plot_search_area=False):
        ploty, left_fitx, right_fitx = self.fit_to_pixels(left_fit, right_fit)

        left_pts = np.transpose(np.vstack((left_fitx, ploty)).astype(np.int32))
        right_pts = np.transpose(np.vstack((right_fitx, ploty)).astype(np.int32))
        cv2.polylines(image, [left_pts, right_pts], False, [255, 255, 0], thickness=3)

        window_img = np.zeros_like(image)
        if plot_search_area:
            line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))

            line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        cv2.addWeighted(image, 1, window_img, 0.3, 0, image)

    def find_window_centroids(self, image):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width) # Create our window template that we will use for convolutions
        shape = image.shape
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*shape[0]/4):,:int(shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-self.window_width/2
        r_sum = np.sum(image[int(3*shape[0]/4):,int(shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-self.window_width/2+int(shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(shape[0]-(level+1)*self.window_height):int(shape[0]-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = self.window_width/2
            l_min_index = int(max(l_center+offset-self.margin,0))
            l_max_index = int(min(l_center+offset+self.margin,shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-self.margin,0))
            r_max_index = int(min(r_center+offset+self.margin,shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids
    
    def window_mask(self, img_ref, center,level):
        output = np.zeros_like(img_ref)
        shape = img_ref.shape
        output[
            int(shape[0]-(level+1)*self.window_height):int(shape[0]-level*self.window_height),
            max(0,int(center-self.window_width/2)):min(int(center+self.window_width/2),
            shape[1])] = 1
        return output
    
    def render_window_centroids(self, image, window_centroids):
        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(image)
            r_points = np.zeros_like(image)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(image, window_centroids[level][0], level)
                r_mask = self.window_mask(image, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | (l_mask == 1)] = 255
                r_points[(r_points == 255) | (r_mask == 1)] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.array(cv2.merge((image,image,image)),np.uint8) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
            return output
        return image
        

def main():
    laneFinding = LaneFinding()
    clip = VideoFileClip("project_video.mp4")
    #clip.duration = 13.0
    processed_clip = clip.fl_image(laneFinding.process_image)
    #processed_clip.preview()
    processed_clip.write_videofile("project_video_processed.mp4", audio=False, threads=4)

    #laneFinding = LaneFinding()
    #clip = VideoFileClip("challenge_video.mp4")
    ##clip.duration = 2.5
    #processed_clip = clip.fl_image(laneFinding.process_image)
    #processed_clip.write_videofile("challenge_video_processed.mp4", audio=False, threads=4)

    #laneFinding = LaneFinding()
    #clip = VideoFileClip("harder_challenge_video.mp4")
    ##clip.duration = 2.3
    #processed_clip = clip.fl_image(laneFinding.process_image)
    #processed_clip.write_videofile("harder_challenge_video_processed.mp4", audio=False, threads=4)

if __name__ == "__main__":
    main()

