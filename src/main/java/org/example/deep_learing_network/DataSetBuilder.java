package org.example.deep_learing_network;

import org.example.ImageCropper;
import org.example.ImageExtractor;
import org.example.PixelValues;
import org.example.image_loader.LoadableImage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * A builder class for creating a {@link DataSet} tailored for machine learning tasks,
 * particularly for image classification.
 *
 * <p>This class uses the builder pattern to allow step-by-step construction of a dataset
 * with configurable parameters such as input size, scaling dimensions, and class encoding.</p>
 */
public class DataSetBuilder {

    private List<LoadableImage> images = new ArrayList<>();
    private int targetPixelRows;
    private int targetPixelCols;
    private int numClasses;
    private boolean includeAlphaChannel = true;

    /**
     * Sets the list of images to be processed for the dataset.
     *
     * @param images A list of {@link LoadableImage} objects containing image data and classifications.
     * @return The current instance of {@link DataSetBuilder} for chaining.
     */
    public DataSetBuilder withImages(List<LoadableImage> images) {
        this.images = images;
        return this;
    }

    /**
     * Sets the target height and width for scaling images.
     *
     * @param rows The target number of rows (height) for the scaled images.
     * @param cols The target number of columns (width) for the scaled images.
     * @return The current instance of {@link DataSetBuilder} for chaining.
     */
    public DataSetBuilder withTargetDimensions(int rows, int cols) {
        this.targetPixelRows = rows;
        this.targetPixelCols = cols;
        return this;
    }

    /**
     * Specifies the number of classes in the dataset, used for one-hot encoding labels.
     *
     * @param numClasses The number of classification labels.
     * @return The current instance of {@link DataSetBuilder} for chaining.
     */
    public DataSetBuilder withNumClasses(int numClasses) {
        this.numClasses = numClasses;
        return this;
    }

    /**
     * Configures whether to include the alpha channel (transparency) in the input features.
     * Defaults to true.
     *
     * @param includeAlpha If true, includes the alpha channel in the feature set.
     * @return The current instance of {@link DataSetBuilder} for chaining.
     */
    public DataSetBuilder includeAlphaChannel(boolean includeAlpha) {
        this.includeAlphaChannel = includeAlpha;
        return this;
    }

    /**
     * Builds the {@link DataSet} based on the configuration provided to the builder.
     *
     * @return A fully constructed {@link DataSet} containing the input features and labels.
     * @throws IllegalStateException if required fields (images, dimensions, or classes) are not set.
     */
    public DataSet build() {
        if (images.isEmpty()) {
            throw new IllegalStateException("Images list cannot be empty. Use withImages() to provide images.");
        }
        if (targetPixelRows <= 0 || targetPixelCols <= 0) {
            throw new IllegalStateException("Target dimensions must be positive. Use withTargetDimensions() to set them.");
        }
        if (numClasses <= 0) {
            throw new IllegalStateException("Number of classes must be positive. Use withNumClasses() to set it.");
        }

        int numExamples = images.size();
        int channels = includeAlphaChannel ? 4 : 3; // 3 for RGB, 4 for RGBA
        int inputSize = targetPixelRows * targetPixelCols * channels;

        // Prepare input and output arrays
        float[][] input = new float[numExamples][inputSize];
        float[][] output = new float[numExamples][numClasses];

        ImageExtractor extractor = new ImageExtractor(targetPixelRows, targetPixelCols);

        for (int i = 0; i < numExamples; i++) {
            System.out.println("Processing image " + i + " of " + numExamples + " to build the dataset");
            List<PixelValues> feature;
            try {
                ImageCropper cropper = new ImageCropper();
                Mat croppedImage = cropper.cropSign(images.get(i).loadMaterial());
                feature = extractor.scaleAndExtractFeaturesFromImage(croppedImage);
            } catch (IllegalStateException e) {
                System.out.println("Could not extract features from image, likely due to cropping error " + images.get(i).path());
                continue;
            }

            // Flatten the features
            float[] flatFeature = new float[inputSize];
            for (int j = 0; j < feature.size(); j++) {
                PixelValues pixel = feature.get(j);
                flatFeature[j * channels] = (float) pixel.getNormalizedRed();
                flatFeature[j * channels + 1] = (float) pixel.getNormalizedGreen();
                flatFeature[j * channels + 2] = (float) pixel.getNormalizedBlue();
                if (includeAlphaChannel) {
                    flatFeature[j * channels + 3] = (float) pixel.getNormalizedAlpha();
                }
            }
            input[i] = flatFeature;

            // One-hot encode the label
            output[i][images.get(i).classification().ordinal()] = 1.0f;
        }

        INDArray inputNDArray = Nd4j.create(input);
        INDArray outputNDArray = Nd4j.create(output);

        return new DataSet(inputNDArray, outputNDArray);
    }
}
