import math
import random
import string
import os

from PIL import Image


def _get_rotated_rect_max_area(w, h, angle):
    """
    Stolen from here: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr

def rotate_max_area(rotated_image, width, height, angle):
    max_width, max_height = _get_rotated_rect_max_area(width,
                                height, math.radians(angle))
    
    w, h = rotated_image.size
    
    y1 = h//2 - int(max_height/2)
    y2 = y1 + int(max_height)
    x1 = w//2 - int(max_width/2)
    x2 = x1 + int(max_width)
    
    return rotated_image.crop(box=(x1,y1,x2,y2))
