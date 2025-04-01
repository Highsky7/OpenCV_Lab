#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
from sensor_msgs.msg import Image  # Import ROS Image message
from cv_bridge import CvBridge, CvBridgeError  # Import CvBridge for conversion

# --- Global variables ---
win_name = 'Camera Access ROS'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
bridge = CvBridge()  # Initialize CvBridge
current_frame = None  # Store the latest frame received from ROS

# --- Image Processing Callback ---
def ImageCallback(ros_image):
    """
    Processing incoming ROS Image messages.
    Updates the global current_frame variable.
    """
    global current_frame
    # rospy.loginfo("Image received!") # Optional: Log message reception

    try:
        # Convert the ROS Image message to an OpenCV image (BGR format)
        # Adjust "bgr8" if your camera publishes in a different encoding (e.g., "rgb8")
        frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return
    
    # Optional: Flip frame if needed (depends on camera mounting)
    # frame = cv2.flip(frame, 1)

    current_frame = frame

# --- Main Program ---
if __name__ == '__main__':
    try:
        # Initialize the ROS node
        # anonymous=True ensures the node has a unique name, avoiding conflicts
        rospy.init_node('ros_image_viewer', anonymous=True)
        rospy.loginfo("ROS Image Viewer Node Started.")

        # --- Get topic name (using a default) ---
        # You could enhance this to read from rospy.get_param or sys.argv if needed
        image_topic = "/camera/color/image_raw" # DEFAULT TOPIC NAME - CHANGE IF YOURS IS DIFFERENT
        rospy.loginfo(f"Subscribing to topic: {image_topic}")

        # --- Create the OpenCV window ---
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) # Create window that can be resized

        # --- Subscribe to the image topic ---
        # Arguments: topic_name, message_type, callback_function, queue_size
        # queue_size=1 means we only process the latest message if we fall behind
        image_subscriber = rospy.Subscriber(image_topic, Image, ImageCallback, queue_size=1)

        # --- Main Loop for Display ---
        # Keep running until ROS is shut down (e.g., Ctrl+C) or ESC is pressed
        while not rospy.is_shutdown():
            # Check if we have received an image frame yet
            if current_frame is not None:
                # Display the current image frame
                cv2.imshow(win_name, current_frame)

            # Wait for a key press for 1 millisecond.
            # The '& 0xFF' is a standard workaround for numlock issues etc. on some systems.
            key = cv2.waitKey(1) & 0xFF

            # Check if the pressed key is ESC (ASCII code 27)
            if key == ord('q') or key == 27:
                rospy.loginfo("ESC key pressed. Shutting down...")
                break # Exit the loop

            # Optional: Add a small sleep if needed, but waitKey(1) usually suffices
            # rospy.sleep(0.01) # Sleep 10ms

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
    finally:
        # --- Cleanup ---
        rospy.loginfo("Closing OpenCV window.")
        cv2.destroyAllWindows() # Destroy the OpenCV display window
        # No need to explicitly release source or shutdown subscriber, ROS handles it