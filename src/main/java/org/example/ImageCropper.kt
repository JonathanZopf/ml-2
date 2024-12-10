package org.example

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class ImageCropper {
    /**
     * Crops the sign from the background, retaining only the pixels inside the largest contour.
     * Crops the image to the bounding rectangle of the largest contour.
     * @param originalSign The original sign image in RGBA format.
     * @return The cropped sign with all pixels outside the largest contour made transparent.
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

        // If no contour is found, return the original sign
        if (largestContour == null) {
            return originalSign
        }

        // Get the bounding rectangle of the largest contour
        val boundingRect = Imgproc.boundingRect(largestContour)

        // Crop the image to the bounding rectangle
        val cropped = Mat(originalSign, boundingRect)

        // Create a mask for the largest contour
        val mask = Mat.zeros(cropped.size(), CvType.CV_8UC1)
        val offsetContour = MatOfPoint()
        val offsetPoints = largestContour.toArray().map {
            Point(it.x - boundingRect.x, it.y - boundingRect.y)
        }
        offsetContour.fromList(offsetPoints)
        Imgproc.drawContours(mask, listOf(offsetContour), -1, Scalar(255.0), Imgproc.FILLED)

        // Apply the mask to the cropped image to make the outside transparent
        val result = Mat.zeros(cropped.size(), CvType.CV_8UC4)
        for (row in 0 until cropped.rows()) {
            for (col in 0 until cropped.cols()) {
                if (mask.get(row, col)[0] > 0) {
                    val pixel = cropped.get(row, col)
                    result.put(row, col, pixel[0], pixel[1], pixel[2], pixel[3])
                }
            }
        }

        return result
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
}
