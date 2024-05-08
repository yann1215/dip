import cv2
import numpy as np


def grey_image_with_circle():
    # Create a blank grey image
    image = np.zeros((256, 256), dtype=np.uint8)

    # Define the center and radius of the circle
    center = (128, 128)
    radius = 50

    # Draw the circle
    cv2.circle(image, center, radius, 255, thickness=-1)  # thickness=-1 fills the circle

    # Display the image
    cv2.imshow("Grey Image with Circle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image
    cv2.imwrite("saved_image/grey_image_with_circle.png", image)


def grey_image_linear_gradient():
    # Define the dimensions of the image
    width = 256
    height = 256

    # Create a blank grey image
    image = np.ones((height, width), dtype=np.uint8) * 255

    # Generate the linear intensity increase
    for x in range(width):
        intensity = int((x / (width - 1)) * 255)  # Linearly increase intensity from 0 to 255
        image[:, x] = intensity

    # Display the image
    cv2.imshow("Grey Image with Linear Gradient", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image
    cv2.imwrite("saved_image/grey_image_linear_gradient.png", image)


grey_image_with_circle()
grey_image_linear_gradient()
