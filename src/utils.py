import cv2
import numpy as np

def get_iou(bbox1, bbox2):

    bb1 = dict()
    bb2 = dict()

    bb1["x1"] = bbox1[0]
    bb1["y1"] = bbox1[1]
    bb1["x2"] = bbox1[2] + bbox1[0]
    bb1["y2"] = bbox1[3] + bbox1[1]

    bb2["x1"] = bbox2[0]
    bb2["y1"] = bbox2[1]
    bb2["x2"] = bbox2[2] + bbox2[0]
    bb2["y2"] = bbox2[3] + bbox2[1]
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def display_overlap(bb1, bb2, overlap_value, img):
    
    # Calcola il centro del bounding box
    center_x = int(bb1[0] + (bb1[2] + bb1[0] - bb1[0]) / 2)
    center_y = int(bb1[1] + (bb1[3] + bb1[1] - bb1[1]) / 2)
    overlap_value = round(overlap_value, 2)

    cv2.rectangle(img,(int(bb1[0]), int(bb1[1])),(int(bb1[2]) + int(bb1[0]), int(bb1[3]) + int(bb1[1])),(18, 122, 43), cv2.FILLED)
    cv2.rectangle(img,(int(bb2[0]), int(bb2[1])),(int(bb2[2]) + int(bb2[0]), int(bb2[3]) + int(bb2[1])),(18, 122, 43), cv2.FILLED)

    cv2.putText(img, str(overlap_value), (center_x - 15, center_y - 5), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def get_polygon_center(polygon: np.ndarray):
    """Calculate the center of a polygon.
    Parameters:
        polygon (np.ndarray): A 2-dimensional numpy ndarray representing the
            vertices of the polygon.
    Returns:
        Point: The center of the polygon, as a tuple with elements x and y.
    """
    shift_polygon = np.roll(polygon, -1, axis=0)
    signed_areas = np.cross(polygon, shift_polygon) / 2
    if signed_areas.sum() == 0:
        center = np.mean(polygon, axis=0).round()
        return (center[0], center[1]) # (X, Y)
    centroids = (polygon + shift_polygon) / 3.0
    center = np.average(centroids, axis=0, weights=signed_areas).round()
    return (center[0], center[1]) # (X, Y)
 
def get_rectangle_center(corner1, corner2):
  """
  Calculate the center point of the rectangle
  """
  center_x = (corner1[0] + corner2[0]) // 2
  center_y = (corner1[1] + corner2[1]) // 2
  return (center_x, center_y)
 
def is_point_inside_polygon(point, polygon):
  """
  Verify if a point is inside a polygon.
 
  Args:
      point (tuple): Coordinates of the point(x, y).
      polygon (np.array): Array of polygon vertex coordinates, in the format np.array([[x1, y1], ..., [xn, yn]]).
 
  Returns:
      bool: True if the point is inside the polygon, False otherwise.
  """
  x, y = point
  n = len(polygon)
  inside = False
 
  for i in range(n):
      x1, y1 = polygon[i]
      x2, y2 = polygon[(i + 1) % n]
 
      if (y1 < y <= y2 or y2 < y <= y1) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
          inside = not inside
 
  return inside