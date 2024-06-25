# Document Scanner

## Overview

The Document Scanner project is a Python application that leverages the OpenCV library to convert images of documents into neatly scanned, perspective-corrected versions. The process involves several stages of image processing, including grayscale conversion, blurring, edge detection, and contour finding. The ultimate goal is to produce a top-down view of the document, similar to a scan from a physical scanner.

## Code Description

The project consists of several key functions, each responsible for a specific step in the image processing pipeline:

### 1) Grayscale Conversion

Converts the input image from color to grayscale using OpenCV’s `cvtColor` function. This simplification helps in subsequent edge detection steps by removing color information.

### 2) Gaussian Blur

Applies a Gaussian blur to the grayscale image to reduce noise and detail. This helps to improve the accuracy of edge detection.

### 3) Edge Detection

Uses the Canny edge detection algorithm to highlight the edges in the blurred image. This step is crucial for identifying the boundaries of the document.

### 4) Edge Dilation

Dilates the edges detected by the Canny algorithm to make them thicker and more pronounced. This enhances the continuity of edges, facilitating contour detection.

### 5) Document Contour Detection

Finds contours in the dilated edge image and sorts them by area. The largest contours are examined to find one that likely represents the document. The contour is approximated to a polygon, and if it has four vertices, it is considered the document's contour.

### 6) Ordering Points

Orders the four vertices of the document contour in a consistent order (top-left, top-right, bottom-right, bottom-left). This is necessary for the perspective transformation.

### 7) Perspective Transformation

Applies a perspective warp to transform the image into a top-down view of the document. The function computes the size of the output image and uses OpenCV’s `getPerspectiveTransform` and `warpPerspective` functions to perform the transformation.

### 8) Main Function

This function orchestrates the entire document scanning process. It loads the image, applies the pre-processing steps, detects edges, finds the document contour, performs perspective transformation, and displays and saves the results.

## Installation

To use the Document Scanner, you need to have Python and OpenCV installed. You can install OpenCV using pip:

```bash
pip install opencv-python

