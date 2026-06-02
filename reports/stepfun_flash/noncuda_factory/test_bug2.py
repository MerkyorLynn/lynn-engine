from bug2 import merge_intervals
def main():
    assert merge_intervals([(1,3),(2,6),(8,10),(15,18)]) == [(1,6),(8,10),(15,18)], "sorted case"
    assert merge_intervals([(1,4),(0,2),(3,5)]) == [(0,5)], "UNSORTED case"
    assert merge_intervals([]) == [], "empty"
    print("PASS")
main()
