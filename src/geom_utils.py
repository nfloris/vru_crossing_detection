import math
import numpy as np
import cv2


def point_on_ellipse(center, axes, angle, theta_deg):
    theta = math.radians(theta_deg)
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))

    x = axes[0] * math.cos(theta)
    y = axes[1] * math.sin(theta)

    x_rot = int(center[0] + (x * cos_a - y * sin_a))
    y_rot = int(center[1] + (x * sin_a + y * cos_a))

    return(x_rot, y_rot)


def get_ellipse_contour(center, axes, angle, start_angle, end_angle):
    points = []

    for theta in range(start_angle, end_angle + 1, 1): 
        pt = point_on_ellipse(center, axes, angle, theta)
        points.append(pt)

    return np.array(points, dtype=np.int32)
    

def is_point_on_ellipse(point, ellipse_contour):
    return cv2.pointPolygonTest(ellipse_contour, point, False) >= 0


