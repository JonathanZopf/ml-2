package org.example.deep_learing_network;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.example.ParametricSigmoid;
import org.nd4j.enums.Mode;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelBuilderWithParametricSigmoid extends ModelBuilder {
    private double alpha;

    public ModelBuilderWithParametricSigmoid withAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }

    @Override
    public MultiLayerNetwork buildAndTrain(org.nd4j.linalg.dataset.DataSet trainingData) {
        Nd4j.getRandom().setSeed(super.seed);
        if (inputSize <= 0) {
            throw new IllegalStateException("Input size must be set and greater than 0. Use withInputSize() to specify it.");
        }
        if (outputSize <= 0) {
            throw new IllegalStateException("Output size must be set and greater than 0. Use withOutputSize() to specify it.");
        }
        if (hiddenLayerConfig == null || hiddenLayerConfig.isEmpty()) {
            throw new IllegalStateException("Hidden layer configuration must be set. Use withHiddenLayerConfig() to specify it.");
        }
        if (alpha <= 0) {
            throw new IllegalStateException("Alpha must be set and greater than 0. Use withAlpha() to specify it.");
        }
        if (hiddenLayerConfig.stream().anyMatch(layer -> layer.getSecond() != Activation.SIGMOID)) {
            throw new IllegalStateException("All hidden layers must use the sigmoid activation function.");
        }


        // Netzwerk-Konfiguration erstellen
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new org.nd4j.linalg.learning.config.Adam(learningRate))
                .list();

        int prevLayerSize = inputSize;

        // Hidden-Layer hinzufügen
        for (var layer : hiddenLayerConfig) {
            listBuilder.layer(new DenseLayer.Builder()
                    .nIn(prevLayerSize)
                    .nOut(layer.getFirst())
                    .activation(new ParametricSigmoid(alpha))
                    .build());
            prevLayerSize = layer.getFirst();
        }

        // Output-Layer hinzufügen
        listBuilder.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(prevLayerSize)
                .nOut(outputSize)
                .activation(outputLayerActivation)
                .build());

        MultiLayerConfiguration config = listBuilder.build();

        // Initialize the model and train
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(logFrequency));

        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingData);
        }

        return model;
    }

    public static ModelBuilderWithParametricSigmoid createFromDefaultModelBuilder(ModelBuilder other) {
        ModelBuilderWithParametricSigmoid modelBuilder = new ModelBuilderWithParametricSigmoid();
        modelBuilder.seed = other.seed;
        modelBuilder.inputSize = other.inputSize;
        modelBuilder.outputSize = other.outputSize;
        modelBuilder.learningRate = other.learningRate;
        modelBuilder.numEpochs = other.numEpochs;
        modelBuilder.logFrequency = other.logFrequency;
        modelBuilder.hiddenLayerConfig = other.hiddenLayerConfig;
        return modelBuilder;
    }
}
