package org.example.deep_learing_network;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * A builder class for constructing and training a {@link MultiLayerNetwork} for machine learning tasks.
 *
 * <p>The builder provides a step-by-step mechanism to configure the neural network's architecture,
 * optimizer, training parameters, and other essential features.</p>
 */
public class ModelBuilder {

    private int inputSize;
    private int outputSize;
    private double learningRate = 0.01;
    private int hiddenLayerSize = 128;
    private int numEpochs = 500;
    private int logFrequency = 10;
    private Activation hiddenLayerActivation = Activation.RELU;
    private Activation outputLayerActivation = Activation.SOFTMAX;

    /**
     * Sets the input size (number of features) for the model.
     *
     * @param inputSize The size of the input layer.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withInputSize(int inputSize) {
        this.inputSize = inputSize;
        return this;
    }

    /**
     * Sets the output size (number of classes) for the model.
     *
     * @param outputSize The size of the output layer.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withOutputSize(int outputSize) {
        this.outputSize = outputSize;
        return this;
    }

    /**
     * Configures the learning rate for the model's optimizer.
     *
     * @param learningRate The learning rate for training.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    /**
     * Sets the number of neurons in the hidden layer.
     *
     * @param hiddenLayerSize The size of the hidden layer.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withHiddenLayerSize(int hiddenLayerSize) {
        this.hiddenLayerSize = hiddenLayerSize;
        return this;
    }

    /**
     * Configures the activation function for the hidden layer.
     *
     * @param activation The activation function to use for the hidden layer.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withHiddenLayerActivation(Activation activation) {
        this.hiddenLayerActivation = activation;
        return this;
    }

    /**
     * Configures the activation function for the output layer.
     *
     * @param activation The activation function to use for the output layer.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withOutputLayerActivation(Activation activation) {
        this.outputLayerActivation = activation;
        return this;
    }

    /**
     * Configures the number of training epochs.
     *
     * @param numEpochs The number of iterations to train the model.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withNumEpochs(int numEpochs) {
        this.numEpochs = numEpochs;
        return this;
    }

    /**
     * Sets the logging frequency for iteration scoring.
     *
     * @param logFrequency The frequency (in iterations) at which the training score is logged.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withLogFrequency(int logFrequency) {
        this.logFrequency = logFrequency;
        return this;
    }

    /**
     * Builds and trains a {@link MultiLayerNetwork} using the specified configuration.
     *
     * @param trainingData The dataset used to train the model.
     * @return A trained {@link MultiLayerNetwork} instance.
     * @throws IllegalStateException if required fields (input size or output size) are not set.
     */
    public MultiLayerNetwork buildAndTrain(DataSet trainingData) {
        if (inputSize <= 0) {
            throw new IllegalStateException("Input size must be set and greater than 0. Use withInputSize() to specify it.");
        }
        if (outputSize <= 0) {
            throw new IllegalStateException("Output size must be set and greater than 0. Use withOutputSize() to specify it.");
        }

        // Build the network configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new org.nd4j.linalg.learning.config.Adam(learningRate)) // Optimizer
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(hiddenLayerSize)
                        .activation(hiddenLayerActivation)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(hiddenLayerSize)
                        .nOut(outputSize)
                        .activation(outputLayerActivation)
                        .build())
                .build();

        // Initialize the network
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(logFrequency));

        // Train the model
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingData);
        }

        return model;
    }
}
