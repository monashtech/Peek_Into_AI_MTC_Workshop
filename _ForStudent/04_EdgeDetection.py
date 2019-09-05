import cv2

def detect_edges(filename_image):
    current_image = cv2.imread(filename_image)
    # using Canny
    current_edges = cv2.Canny(
        current_image,
        threshold1 = 200,
        threshold2 = 300
    )
    return current_edges

def show_edges(current_edges):
    cv2.imshow("Showing the edges found...", current_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # set to current working directory
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # filename to detect the edges
    filename_image = "cat_01.jpg"

    # get the edges
    current_edges = detect_edges(filename_image)
    # show the edges
    show_edges(current_edges)

    
    
