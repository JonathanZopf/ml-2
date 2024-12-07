package org.example

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Crops the sign from the background and makes the background transparent.
 * @param originalSign The original sign image in RGBA format.
 * @return The cropped sign with a transparent background
 */
fun cropSign(originalSign: Mat): Mat {
    // Convert the image to binary based on the grayscale
    val gray = Mat()
    Imgproc.cvtColor(originalSign, gray, Imgproc.COLOR_RGBA2GRAY)

    val contours = getAllContoursAdapting(gray, 100.0, 200.0)

    // Find the largest contour
    var largestContour: MatOfPoint? = null
    var largestArea = Double.MIN_VALUE
    for (contour in contours) {
        val area = Imgproc.contourArea(contour)
        if (area > largestArea) {
            largestArea = area
            largestContour = contour
        }
    }

    // Instead of throwing an exception (like it would be preferable in a production system)
    // return the original sign if no contour is found to continue the flow of the program
    if (largestContour == null) {
        return originalSign
    }

    // Get the convex hull of the largest contour to complete the contour
    val convexHull = MatOfPoint()
    val hullIndices = MatOfInt()
    Imgproc.convexHull(largestContour, hullIndices)
    val hullPoints = hullIndices.toArray().map { largestContour.toArray()[it] }
    convexHull.fromList(hullPoints)

    return getSignWithOutsideTransparent(originalSign, convexHull)
}

/**
 * Recursively finds all contours of a sign by adapting the Canny edge detection thresholds.
 * @param grayscaleSign The sign in grayscale format.
 * @param threshold1 The first threshold for the Canny edge detection.
 * @param threshold2 The second threshold for the Canny edge detection.
 * @return A list of all contours of the sign.
 * @throws IllegalStateException If no contour is found even after adapting the thresholds.
 */
private fun getAllContoursAdapting(grayscaleSign: Mat, threshold1: Double, threshold2: Double): List<MatOfPoint> {
    if (threshold1 < 1.0 || threshold2 < 1.0) {
        throw IllegalStateException("No contour found")
    }
    // Apply Canny edge detection
    val edges = Mat()
    Imgproc.Canny(grayscaleSign, edges, threshold1, threshold2)

    // Find contours
    val contours = ArrayList<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

    if (contours.isEmpty()) {
        return getAllContoursAdapting(grayscaleSign, threshold1 / 1.5, threshold2 / 1.5)
    }
    return contours
}

/**
 * Makes all pixels outside the convex hull transparent and returns the modified image.
 * Converts the image to BGRA format.
 * @param sign The original sign image in RGBA format.
 * @param hull The convex hull that defines the sign.
 * @return The modified image with a transparent background.
 */
private fun getSignWithOutsideTransparent(sign: Mat, hull: MatOfPoint): Mat {
    // Create a mask based on the convex hull
    val mask = Mat.zeros(sign.size(), CvType.CV_8UC1)
    Imgproc.drawContours(mask, listOf(hull), -1, Scalar(255.0), Imgproc.FILLED)

    for (row in 0 until sign.rows()) {
        for (col in 0 until sign.cols()) {
            val pixelMask = mask.get(row, col)[0]
            if (pixelMask == 0.0) { // Outside the hull
                sign.put(row, col, 0.0, 0.0, 0.0, 0.0) // Make transparent
            }
        }
    }

    return sign
}
