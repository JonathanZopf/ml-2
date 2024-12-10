package org.example;

import nu.pattern.OpenCV;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.example.deep_learing_network.DataSetBuilder;
import org.example.deep_learing_network.Evaluator;
import org.example.deep_learing_network.ModelBuilder;
import org.example.image_loader.ImageLoader;
import org.example.image_loader.ImageLoaderResult;
import org.example.image_loader.LoadableImage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.opencv.core.Core;

/**
 * Main class for training and evaluating a deep learning model for sign classification.
 */
public class Main {
    final static int SCALE_TARGET_PIXEL_SIZE_ROWS = 60;
    final static int SCALE_TARGET_PIXEL_SIZE_COLS = 80;

    public static void main(String[] args) {
        int imagesForTraining = 10;
        int imagesForTesting = 10;

        // Initialize OpenCV
        OpenCV.loadShared();

        // Load Images
        ImageLoader loader = new ImageLoader(imagesForTraining, imagesForTesting);
        ImageLoaderResult loaderResult = loader.loadImages();

        for (LoadableImage image : loaderResult.imagesForTraining()) {
            System.out.println(image.path());
        }

        // Prepare Training and Testing Data
        // DefaultDataSetBuilder is shared between training and testing data because most of the parameters are the same
        DataSetBuilder defaultDataSetBuilder = new DataSetBuilder();
        defaultDataSetBuilder.withTargetDimensions(SCALE_TARGET_PIXEL_SIZE_ROWS, SCALE_TARGET_PIXEL_SIZE_COLS);
        defaultDataSetBuilder.withNumClasses(SignClassification.values().length);

        DataSet trainingData = defaultDataSetBuilder.withImages(loaderResult.imagesForTraining()).build();
        DataSet testingData = defaultDataSetBuilder.withImages(loaderResult.imagesForTesting()).build();

        // Define and Train the Neural Network
        MultiLayerNetwork model = new ModelBuilder()
                .withInputSize(SCALE_TARGET_PIXEL_SIZE_ROWS * SCALE_TARGET_PIXEL_SIZE_COLS * 4)
                .withOutputSize(SignClassification.values().length)
                .withLearningRate(0.01)
                .withHiddenLayerSize(128)
                .withHiddenLayerActivation(Activation.RELU)
                .withOutputLayerActivation(Activation.SOFTMAX)
                .withNumEpochs(100)
                .withLogFrequency(10)
                .buildAndTrain(trainingData);

        // Evaluate the Model
        Evaluator evaluator = new Evaluator(model, testingData);
        evaluator.evaluateModel();
    }
}
