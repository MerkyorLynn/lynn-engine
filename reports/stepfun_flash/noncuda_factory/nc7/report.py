from geom import rectangle_area
def describe(w, h):
    return "rect %sx%s area=%s" % (w, h, rectangle_area({'width': w, 'height': h}))
