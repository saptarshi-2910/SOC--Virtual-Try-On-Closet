import cv2
import numpy as np

#Converting the input image to grayscale.
def to_grayscale(image):    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Applying Gaussian blur to the input image.
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Applying Canny edge detection to the input image.
def detect_edges(image, lower_threshold=75, upper_threshold=75):
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

# Dilating the edges to make them thicker.
def dilate_edges(edges, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    return dilated

#  Finding the contour that most likely represents the document.
def find_document_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter out smaller contours
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                return approx
    return None

# Order the points in a consistent order: top-left, top-right, bottom-right, bottom-left.
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

#Perform a perspective warp to transform the image to a top-down view.
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def scan_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image '{image_path}'.")
        return

    # Applying pre-processing functions
    grayscale = to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale)
    edges = detect_edges(blurred)
    dilated_edges = dilate_edges(edges)

    # Finding the document contour
    document_contour = find_document_contour(dilated_edges)

    # Drawing the contour on the original image if found
    if document_contour is not None:
        cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 2)
        warped = four_point_transform(image, document_contour.reshape(4, 2))
        cv2.imshow("Warped", warped)
        cv2.imwrite("scanned_document.jpg", warped)  # Save the scanned document

    # Displaying the processed images
    cv2.imshow("Original", image)
    cv2.imshow("Edges", edges)
    cv2.imshow("Dilated Edges", dilated_edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "C:\\Users\\Hardik Gohil\\Downloads\\HELP.jpg"
    scan_image(image_path)
