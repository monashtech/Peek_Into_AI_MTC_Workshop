# import the openCV library
import cv2

def detect_eye(filename_image, filename_haar="haarcascade_eye.xml"):
    # load the image and convert it to grayscale
    current_image = cv2.imread(filename_image)
    current_grayscale = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # load the haarcascade
    current_cascade = cv2.CascadeClassifier(filename_haar)

    # perform the detection
    current_eyes = current_cascade.detectMultiScale(
        current_grayscale,
        # note: we can provide no parameters and it woul still work
        scaleFactor = 1.03,
        minNeighbors = 5,
        minSize = (20,10),
        maxSize = (50,30)
    )

    # return the results with the current image
    return current_image, current_eyes
    
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
    filename_image = "Ellen.jpg"
    filename_haar_eye = "haarcascade_eye.xml"

    # detect the faces
    current_image, current_eyes = detect_eye(filename_image, filename_haar_eye)
    # print number of eyes
    print("Found {0} eyes!".format(len(current_eyes)))
    # draw out the image
    draw_detection(current_image, current_eyes)