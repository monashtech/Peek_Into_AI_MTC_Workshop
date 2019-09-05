# import the openCV library
import cv2

def detect_face(filename_image, filename_haar="haarcascade_frontalface_default.xml"):
    # load the image and convert it to grayscale
    current_image = cv2.imread(filename_image)
    current_grayscale = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # load the haarcascade
    current_cascade = cv2.CascadeClassifier(filename_haar)

    # perform the detection
    current_faces = current_cascade.detectMultiScale(
        current_grayscale,
        scaleFactor = 1.05,
        minNeighbors = 1,
        minSize = (50, 30),
        # scaleFactor = the size of the face in the image that is detectable.
        # minNeighbors = higher value means less detection but better detection. usually 3-6 but the proper way is to do it empirically
        # minSize = minimum size for the face to be detected. smaller objects are ignored. usually 30, 30
        # note: it is a tuple for width-height
        # maxSize = maximum possible object size
        # note: we usually don't set this
        # minsize is tuple in the width-length
    )

    # return the results with the current image
    return current_image, current_faces
    
def draw_detection(current_image, current_boxes):
    # draw the boxes
    for (x, y, w, h) in current_boxes:
        cv2.rectangle(
            current_image,
            (x, y),
            (x+w, y+h),
            color = (0, 255, 0),
            thickness = 2
        )

    cv2.imshow("Showing detection", current_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # set to current working directory
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # the filenames
    filename_image = "family.jpg"
    filename_haar_face = "haarcascade_frontalface_default.xml"

    # detect the faces
    current_image, current_faces = detect_face(filename_image, filename_haar_face)
    # print number of faces
    print("Found {0} faces!".format(len(current_faces)))
    # draw out the image
    draw_detection(current_image, current_faces)