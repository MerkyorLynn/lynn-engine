from geom import rectangle_area
from report import describe
def main():
    assert rectangle_area({'width': 3, 'height': 4}) == 12
    assert describe(2, 5) == "rect 2x5 area=10"
    print("PASS")
main()
