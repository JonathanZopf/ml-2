package org.example.deep_learing_network;

import kotlin.Pair;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

/**
 * A builder class for constructing and training a {@link MultiLayerNetwork} for machine learning tasks.
 *
 * <p>The builder provides a step-by-step mechanism to configure the neural network's architecture,
 * optimizer, training parameters, and other essential features.</p>
 */
public class ModelBuilder {

    protected int inputSize;
    protected int outputSize;
    protected double learningRate = 0.01;
    protected List<Pair<Integer, Activation>> hiddenLayerConfig;
    protected int numEpochs = 500;
    protected int logFrequency = 10;
    protected Activation outputLayerActivation = Activation.SOFTMAX;
    protected long seed = 1; // Default seed


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
     * Sets the configuration for the hidden layers of the model.
     * Each pair represents the size of the layer and the activation function to use.
     * The first element of the list represents the first hidden layer, the second element the second hidden layer, and so on.
     * The last element of the list represents the last hidden layer before the output layer.
     */
    public ModelBuilder withHiddenLayerConfig(List<Pair<Integer, Activation>> hiddenLayerSize) {
        this.hiddenLayerConfig = hiddenLayerSize;
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
     * Sets the random seed for reproducibility.
     *
     * @param seed The seed value for random number generation.
     * @return The current instance of {@link ModelBuilder} for chaining.
     */
    public ModelBuilder withSeed(long seed) {
        this.seed = seed;
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
        Nd4j.getRandom().setSeed(seed);
        if (inputSize <= 0) {
            throw new IllegalStateException("Input size must be set and greater than 0. Use withInputSize() to specify it.");
        }
        if (outputSize <= 0) {
            throw new IllegalStateException("Output size must be set and greater than 0. Use withOutputSize() to specify it.");
        }
        if (hiddenLayerConfig == null || hiddenLayerConfig.isEmpty()) {
            throw new IllegalStateException("Hidden layer configuration must be set. Use withHiddenLayerConfig() to specify it.");
        }

        // Netzwerk-Konfiguration erstellen
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed) // Seed für die Netzwerk-Konfiguration
                .updater(new org.nd4j.linalg.learning.config.Adam(learningRate))
                .list();

        int prevLayerSize = inputSize; // Start mit der Eingabegröße

        // Hidden-Layer hinzufügen
        for (var layer : hiddenLayerConfig) {
            listBuilder.layer(new DenseLayer.Builder()
                    .nIn(prevLayerSize) // Größe der vorherigen Schicht
                    .nOut(layer.getFirst())    // Größe der aktuellen Schicht
                    .activation(layer.getSecond())
                    .build());
            prevLayerSize = layer.getFirst(); // Update für die nächste Schicht
        }

        // Output-Layer hinzufügen
        listBuilder.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(prevLayerSize) // Größe der letzten Hidden-Layer
                .nOut(outputSize)  // Anzahl der Ausgabeklassen
                .activation(outputLayerActivation)
                .build());


        MultiLayerConfiguration config = listBuilder.build();

        // Netzwerk initialisieren
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
