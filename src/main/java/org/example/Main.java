package org.example;

import nu.pattern.OpenCV;
import org.example.image_loader.ImageLoader;
import org.example.image_loader.ImageLoaderResult;
import org.example.image_loader.LoadableImage;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.Color;
import java.io.*;
import java.util.List;


public class Main {
    final static int SCALE_TARGET_PIXEL_SIZE_ROWS = 60;
    final static int SCALE_TARGET_PIXEL_SIZE_COLS = 80;
    public static void main(String[] args) {
        int imagesForTraining = 100;
        int imagesForTesting = 50;

        OpenCV.loadShared();
        ImageLoader loader = new ImageLoader(imagesForTraining, imagesForTesting);
        ImageLoaderResult loaderResult = loader.loadImages();

        ImageExtractor extractor = new ImageExtractor(SCALE_TARGET_PIXEL_SIZE_ROWS, SCALE_TARGET_PIXEL_SIZE_COLS);
        List<List<Color>> features = extractor.scaleAndExtractFeaturesFromImages(loaderResult.imagesForTesting().stream().map(it->ImageCropperKt.cropSign(it.loadMaterial())).toList());

        for (List<Color> feature : features) {
            writeToFileForDebug(feature);
        }
    }

    private static void writeToFileForDebug(List<Color> features) {
        Mat newMat = Mat.zeros(SCALE_TARGET_PIXEL_SIZE_ROWS, SCALE_TARGET_PIXEL_SIZE_COLS, CvType.CV_8UC4);
        for (int i = 0; i < SCALE_TARGET_PIXEL_SIZE_ROWS; i++) {
            for (int j = 0; j < SCALE_TARGET_PIXEL_SIZE_COLS; j++) {
                Color color = features.get(i * SCALE_TARGET_PIXEL_SIZE_COLS + j);
                newMat.put(i, j, new double[]{color.getRed(), color.getGreen(), color.getBlue(), color.getAlpha()});
            }
        }

        try {
            ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
            InputStream fileLocationFile = classLoader.getResourceAsStream("file_paths/debug_output.txt");
            if (fileLocationFile == null) {
                throw new IllegalArgumentException("There is no file_paths/debug_output.txt in the resources folder");
            }

            StringBuilder debugProcessedFileLocation = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fileLocationFile))) {
                debugProcessedFileLocation.append(reader.readLine());
            }

            debugProcessedFileLocation.append(newMat.hashCode()).append(".jpg");

            // Check if the directory exists
            String directoryPath = debugProcessedFileLocation.toString().substring(0, debugProcessedFileLocation.lastIndexOf("/"));
            File directory = new File(directoryPath);
            if (!directory.exists()) {
                throw new IllegalArgumentException("Invalid directory: " + debugProcessedFileLocation);
            }

            // Write the image to the file
            Imgproc.cvtColor(newMat, newMat, Imgproc.COLOR_RGBA2BGR);
            Imgcodecs.imwrite(debugProcessedFileLocation.toString(), newMat);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}