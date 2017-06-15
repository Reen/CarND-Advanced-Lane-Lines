# Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Distorted vs. Undistorted"
[image2]: ./output_images/orig_vs_undistorted.png "Road Transformed"
[image3]: ./output_images/color_masking.png "Color Masking"
[image3a]: ./output_images/undistorted_vs_mask.png "Binary Example"
[image4]: ./output_images/source_warped_points.png "Warp Example"
[image5]: ./output_images/fitting.png "Fit Visual"
[image6]: ./output_images/output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./camera_calibration.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  I use `cv2.cornerSubPix()` to refine the pixel positions to sub-pixel level.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Distorted vs. Undistorted][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I applied the `cv2.undistort()` function to one of the provided test images:
![Distorted real-world image vs Undistorted][image2]
All image processing code, including the undistortion is located in its own class `ImageProcessing` within `LaneFinding.py`. A separate class holds the `CameraCalibration` and takes care of deserializing the pre-calcualted calibration.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 102 through 111 in `LaneFinding.py`).  Here's an example of my output for this step.

![A selection of test images with the resulting color mask and sobel mask.][image3]

The final mask looks like this:
![Undistorted vs. Mask Image.][image3a]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is located in the method `ImageProcessing::perspective_transform`, which appears in lines 124 in the file `LaneFinding.py`.  The `perspective_transform()` function takes as inputs an image (`img`). The source and destination points are members of the class `ImageProcessing`. I chose the hardcode the source and destination points in the following manner:

```python
source_points = np.array([(608, 440), (203, 720), (1127, 720), (676, 440)], dtype=np.float32)
warped_points = np.array([(320, 0), (320, 720), (960, 720), (960, 0)], dtype=np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 608, 440      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 676, 440      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the examples provided in the lesson, using a sliding window fit (see `LaneFinding::sliding_window_fit` in `LaneFinding.py`, line #)
to find an initial estimate for left and right lane, followed by a re-fit procedure (see `LaneFinding::refit` in `LaneFinding.py`, line #).
To fit a polynom through the extracted pixels, I use a rotated and shifted coordinate system.
As in the lesson, I fit a second order polynomial w.r.t. _y_, i.e. _f(y)_ instead of _f(x)_, but moved the origin of the coordinate system to the lower left pixel, with
_x_ pointing to the right and _y_ pointing upwards. This way, using the equation _f(y) = a*x^2 + b*x + c_, we end up having a parameter _c_ describing
the offset of the lane from the left border, a parameter _b ≈ 0_ being the slope and _a_ being the curvature.

![Results of the sliding window fit and the re-fit procedure][image5]

Looking at professional systems like MobilEye, one finds that it is even possible to use third-order polynomial equation.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated within the `Line` class and is updated each time a new fit is processed (see line # of `LaneFinding.py`).
The offset is calculated in `LaneFinding::calculate_offset` (line # of `LaneFinding.py`). The calculated offset is calculated in the automotive coordiate system with the _x_-axis pointing forward and _y_ pointing to the left.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `LaneFinding.py` in the function `Lanefinding::process_image()`.  Here is an example of my result on a test image:

![Output image with curvature and offset rendered and the fitted lanes overlayed][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
