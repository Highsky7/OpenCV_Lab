#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy
from sensor_msgs.msg import Image # Import ROS Image message
from cv_bridge import CvBridge, CvBridgeError # Import CvBridge for conversion

# --- Filter constants ---
PREVIEW  = 0  # Preview Mode
BLUR     = 1  # Blurring Filter(used to smoothing images that can be robust to noises from raw data)
FEATURES = 2  # Corner Feature Detector(minDistance, maxCorners and qualityLevel(threshold = maxCorners*qualityLevel) is important)
CANNY    = 3  # Canny Edge Detector

# --- Global variables ---
image_filter = PREVIEW
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)
win_name = "Camera Filters ROS"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
bridge = CvBridge() # Initialize CvBridge
current_frame = None # Store the latest frame received from ROS
# alive = True # Controls the main display loop

# --- Image Processing Callback ---
def image_callback(ros_image):
    """
    Processes incoming ROS Image messages.
    Converts the ROS Image to an OpenCV image and applies the selected filter.
    Updates the global current_frame variable.
    """
    global current_frame, image_filter, feature_params

    # rospy.loginfo("Image received!") # Optional: Log message reception

    try:
        # Convert the ROS Image message to an OpenCV image (BGR format)
        # Adjust "bgr8" if your camera publishes in a different encoding (e.g., "rgb8")
        frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return # Skip processing this frame if conversion fails

    # Optional: Flip frame if needed (depends on camera mounting)
    # frame = cv2.flip(frame, 1)

    result = None # Initialize result for this frame

    # --- Apply selected filter ---
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        # Ensure frame is 8-bit single channel for Canny if needed,
        # but Canny often works directly on BGR (internally converts)
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Optional pre-conversion
        result = cv2.Canny(frame, 80, 150) # (frame, lower threshold, upper threshold) (if we set lower threshold much lower, allowing strong edges to be connected with weaker edges->get much connected edges)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13)) # (frame, kernel size)
    elif image_filter == FEATURES:
        result = frame.copy() # Work on a copy to draw circles
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            # Ensure corners are correctly formatted for drawing
            corners_int = numpy.int0(corners) # Convert to integer for drawing
            for corner in corners_int:
                 x, y = corner.ravel() # Flatten the array
                 cv2.circle(result, (x, y), 10, (0, 255, 0), 1) # Draw green circle

    # Update the global variable to be displayed by the main loop
    if result is not None:
        current_frame = result

# --- Main Execution ---
if __name__ == '__main__':
    rospy.init_node('camera_filter_node', anonymous=True)
    rospy.loginfo("Camera Filter Node Started.")

    # --- Subscribe to the ROS topic ---
    # Replace '/camera/color/image_raw' if your topic name is different
    topic_name = "/camera/color/image_raw/" # Example topic name
    image_sub = rospy.Subscriber(topic_name, Image, image_callback, queue_size=1)
    rospy.loginfo(f"Subscribed to {topic_name}")

    # --- Main Loop for Display and Keyboard Input ---
    # The callback handles image processing. This loop handles display and user interaction.
    try:
        # while alive and not rospy.is_shutdown():
        while not rospy.is_shutdown():
            if current_frame is not None:
                cv2.imshow(win_name, current_frame)

            key = cv2.waitKey(1) & 0xFF # Get key press (wait 1ms), mask for 64-bit systems

            if key == ord('q') or key == 27: # Quit on 'q' or ESC
                rospy.loginfo("Shutdown requested.")
                # alive = False
                break # Exit the loop immediately
            elif key == ord('c'):
                rospy.loginfo("Filter changed to: CANNY")
                image_filter = CANNY
            elif key == ord('b'):
                rospy.loginfo("Filter changed to: BLUR")
                image_filter = BLUR
            elif key == ord('f'):
                rospy.loginfo("Filter changed to: FEATURES")
                image_filter = FEATURES
            elif key == ord('p'):
                rospy.loginfo("Filter changed to: PREVIEW")
                image_filter = PREVIEW

            # Optional: Add a small sleep if waitKey(1) is too CPU intensive,
            # but usually it's sufficient.
            # rospy.sleep(0.01) # Sleep for 10ms

    except Exception as e:
         rospy.logerr(f"An error occurred in the main loop: {e}")
    finally:
        # --- Cleanup ---
        rospy.loginfo("Shutting down.")
        cv2.destroyAllWindows() # Close OpenCV windows
        # No need to release source, ROS handles the subscription shutdown.

    # Note: rospy.spin() is not used here because we need the main thread
    # to run the cv2.waitKey() and display loop. The subscriber runs
    # in its own thread managed by rospy.