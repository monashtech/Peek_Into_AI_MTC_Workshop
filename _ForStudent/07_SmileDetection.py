# import the openCV library
import cv2

def detect_smile(filename_image, filename_haar="haarcascade_smile.xml"):
    # load the image and convert it to grayscale
    current_image = cv2.imread(filename_image)
    current_grayscale = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # load the haarcascade
    current_cascade = cv2.CascadeClassifier(filename_haar)

    # perform the detection
    current_smiles = current_cascade.detectMultiScale(
        current_grayscale,
        scaleFactor = 1.8,
        minNeighbors = 8
    )

    # return the results with the current image
    return current_image, current_smiles
    
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
    filename_haar_smile = " "

    # detect the faces
    current_image, current_smiles = detect_smile(filename_image, filename_haar_smile)
    # print number of smiles
    print("Found {0} smiles!".format(len(current_smiles)))
    # draw out the image
    draw_detection(current_image, current_smiles)