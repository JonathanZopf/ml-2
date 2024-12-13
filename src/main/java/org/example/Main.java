package org.example;

import nu.pattern.OpenCV;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.example.deep_learing_network.DataSetBuilder;
import org.example.deep_learing_network.Evaluator;
import org.example.deep_learing_network.ModelBuilder;
import org.example.image_loader.ImageLoader;
import org.example.image_loader.ImageLoaderResult;
import org.example.image_loader.LoadableImage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Main class for training and evaluating a deep learning model for sign classification.
 */
public class Main {
    private static final int SCALE_TARGET_PIXEL_SIZE_ROWS = 60;//60
    private static final int SCALE_TARGET_PIXEL_SIZE_COLS = 80;//80
    private static final boolean REGENERATE_DATA = false; // Set to false to load datasets from disk

    public static void main(String[] args) {
        //int imagesForTraining = 10000;
        //int imagesForTesting = 1000;
        int imagesForTraining = 35000;
        int imagesForTesting = 3500;

        // Initialize OpenCV
        OpenCV.loadShared();

        // Load file paths from an external resource file
        String parentFolderLocation = loadParentFolderLocation("file_paths/dataset_paths.txt");
        File trainingDataFile = new File(parentFolderLocation, "trainingData.bin");
        File testingDataFile = new File(parentFolderLocation, "testingData.bin");

        DataSet trainingData;
        DataSet testingData;

        if (REGENERATE_DATA) {
            // Load Images
            ImageLoader loader = new ImageLoader(imagesForTraining, imagesForTesting);
            ImageLoaderResult loaderResult = loader.loadImages();

            for (LoadableImage image : loaderResult.imagesForTraining()) {
                System.out.println(image.path());
            }

            // Prepare Training and Testing Data
            DataSetBuilder defaultDataSetBuilder = new DataSetBuilder();
            defaultDataSetBuilder.withTargetDimensions(SCALE_TARGET_PIXEL_SIZE_ROWS, SCALE_TARGET_PIXEL_SIZE_COLS);
            defaultDataSetBuilder.withNumClasses(SignClassification.values().length);

            trainingData = defaultDataSetBuilder.withImages(loaderResult.imagesForTraining()).build();
            testingData = defaultDataSetBuilder.withImages(loaderResult.imagesForTesting()).build();

            // Save datasets to disk
            trainingData.save(trainingDataFile);
            testingData.save(testingDataFile);
            System.out.println("Datasets saved to disk at: " + parentFolderLocation);
        } else {
            // Load datasets from disk
            trainingData = new DataSet();
            trainingData.load(trainingDataFile);
            testingData = new DataSet();
            testingData.load(testingDataFile);
            System.out.println("Datasets loaded from disk at: " + parentFolderLocation);
        }

        // Define and Train the Neural Network
        List<Integer> hiddenLayerSize = Arrays.asList(500, 250, 128, 64);
        MultiLayerNetwork model = new ModelBuilder()
                .withInputSize(SCALE_TARGET_PIXEL_SIZE_ROWS * SCALE_TARGET_PIXEL_SIZE_COLS * 4)
                .withOutputSize(SignClassification.values().length)
                .withLearningRate(0.0001)
                .withHiddenLayerSizes(hiddenLayerSize)
                .withHiddenLayerActivation(Activation.RELU)
                .withOutputLayerActivation(Activation.SOFTMAX)
                .withNumEpochs(100)
                .withLogFrequency(10)
                .buildAndTrain(trainingData);



        // Evaluate the Model
        Evaluator evaluator = new Evaluator(model, testingData);
        evaluator.evaluateModel();


    }

    /**
     * Loads the parent folder location from the specified file.
     *
     * @param resourcePath Path to the resource file containing the parent folder location.
     * @return The parent folder location as a string.
     */
    private static String loadParentFolderLocation(String resourcePath) {
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        InputStream fileLocationStream = classloader.getResourceAsStream(resourcePath);

        if (fileLocationStream == null) {
            throw new IllegalArgumentException("There is no " + resourcePath + " in the resources folder");
        }

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(fileLocationStream))) {
            return reader.lines().collect(Collectors.joining("\n")).trim();
        } catch (IOException e) {
            throw new RuntimeException("Error reading " + resourcePath, e);
        }
    }
}