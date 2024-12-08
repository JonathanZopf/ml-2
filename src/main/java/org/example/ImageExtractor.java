package org.example;

import org.example.PixelValues;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ImageExtractor {
    private final int targetRows;
    private final int targetCols;

    public ImageExtractor(int targetRows, int targetCols) {
        this.targetRows = targetRows;
        this.targetCols = targetCols;
    }

    public List<PixelValues> scaleAndExtractFeaturesFromImage(Mat image) {
        Mat rescaledImage = rescaleImage(image);
        return extractFeaturesFromImage(rescaledImage);
    }

    private Mat rescaleImage(Mat input) {
        Mat resized = new Mat();
        Size targetSize = new Size(targetCols, targetRows);
        Imgproc.resize(input, resized, targetSize);
        return resized;
    }

    private List<PixelValues> extractFeaturesFromImage(Mat image) {
        List<PixelValues> res = new ArrayList<>();
        for (int row = 0; row < image.rows(); row++) {
            for (int col = 0; col < image.cols(); col++) {
                double[] pixel = image.get(row, col);
                if (pixel == null) {
                    throw new IllegalArgumentException("Pixel is null");
                }
                if (pixel.length != 4) {
                    throw new IllegalArgumentException("Pixel does not have 4 channels");
                }
                res.add(PixelValues.fromUnormalizedArray(pixel));
            }
        }
        return res;
    }
}
