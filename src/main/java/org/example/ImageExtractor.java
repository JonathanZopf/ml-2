package org.example;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class ImageExtractor {
    private final int targetRows;
    private final int targetCols;

    public ImageExtractor(int targetRows, int targetCols) {
        this.targetRows = targetRows;
        this.targetCols = targetCols;
    }

    public List<List<Color>> scaleAndExtractFeaturesFromImages(List<Mat> images) {
        List<Mat> rescaledImages = rescaleImages(images);
        List<List<Color>> extractionResult = rescaledImages.stream()
                .map(this::extractFeaturesFromImage)
                .toList();

        boolean hasDifferentSizes = extractionResult.stream()
                .map(List::size)
                .distinct()
                .count() > 1;

        if (hasDifferentSizes) {
            throw new IllegalArgumentException("Images have different sizes!");
        }

        return extractionResult;
    }

    private List<Mat> rescaleImages(List<Mat> input) {
        return input.stream().map(image -> {
            Mat resized = new Mat();
            Size targetSize = new Size(targetCols, targetRows);
            Imgproc.resize(image, resized, targetSize);
            return resized;
        }).toList();
    }

    private List<Color> extractFeaturesFromImage(Mat image) {
        List<Color> colors = new ArrayList<>();
        for (int row = 0; row < image.rows(); row++) {
            for (int col = 0; col < image.cols(); col++) {
                double[] pixel = image.get(row, col);
                if (pixel == null) {
                    throw new IllegalArgumentException("Pixel is null");
                }
                if (pixel.length != 4) {
                    throw new IllegalArgumentException("Pixel does not have 4 channels");
                }
                Color color = new Color(
                        (int) pixel[0],
                        (int) pixel[1],
                        (int) pixel[2],
                        (int) pixel[3]);
                colors.add(color);
            }
        }
        return colors;
    }
}
