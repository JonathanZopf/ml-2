package org.example.deep_learing_network;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.example.SignClassification;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The Evaluator class is designed to evaluate the performance of a trained neural network model
 * using a provided testing dataset. This class implements the {@link DataSetIterator} interface
 * to act as an iterator for the testing data, making it compatible with Deeplearning4j's evaluation framework.
 */
public class Evaluator implements DataSetIterator {

    /**
     * The trained neural network model to be evaluated.
     */
    private MultiLayerNetwork model;

    /**
     * The dataset used for testing the model's performance.
     */
    private DataSet testingData;

    /**
     * A flag to indicate whether the testing data has more elements to iterate over.
     * This is initially set to true and set to false after the first call to {@code next()}.
     */
    private boolean hasNext = true;

    /**
     * Constructs an Evaluator instance with a given trained model and testing dataset.
     *
     * @param model       The trained {@link MultiLayerNetwork} model to be evaluated.
     * @param testingData The {@link DataSet} containing testing data.
     */
    public Evaluator(MultiLayerNetwork model, DataSet testingData) {
        this.model = model;
        this.testingData = testingData;
    }

    /**
     * Evaluates the performance of the neural network model on the provided testing dataset.
     * The evaluation metrics are printed to the standard output.
     */
    public void evaluateModel() {
        var eval = model.evaluate(this); // Uses the current Evaluator instance as a DataSetIterator
        System.out.println(eval.stats()); // Prints evaluation statistics such as accuracy and F1 score
    }

    /**
     * Evaluates the performance of the neural network model on the provided testing dataset.
     * @return An Evaluation object of the framework
     */
    public Evaluation getEvaluationResult() {
        return model.evaluate(this);
    }

    /**
     * Checks whether there are more data batches to iterate over.
     *
     * @return {@code true} if there is at least one more batch; {@code false} otherwise.
     */
    @Override
    public boolean hasNext() {
        return hasNext;
    }

    /**
     * Returns the next batch of data. In this implementation, it will only return the testing dataset once.
     *
     * @return The {@link DataSet} for testing.
     */
    @Override
    public DataSet next() {
        hasNext = false; // Ensures the iterator only returns data once
        return testingData;
    }

    /**
     * Returns the next batch of data with a specified batch size.
     * Note: In this implementation, it ignores the batch size and returns the full testing dataset.
     *
     * @param num The requested batch size (ignored in this implementation).
     * @return The {@link DataSet} for testing.
     */
    @Override
    public DataSet next(int num) {
        return next(); // Redirects to the single-batch implementation
    }

    /**
     * Returns the number of input features in the dataset.
     *
     * @return The number of input columns (features) in the dataset.
     */
    @Override
    public int inputColumns() {
        return testingData.getFeatures().columns();
    }

    /**
     * Returns the number of output labels (classes) in the dataset.
     *
     * @return The number of label columns in the dataset.
     */
    @Override
    public int totalOutcomes() {
        return testingData.getLabels().columns();
    }

    /**
     * Indicates whether the iterator supports resetting.
     *
     * @return {@code false} as resetting is not supported in this implementation.
     */
    @Override
    public boolean resetSupported() {
        return false;
    }

    /**
     * Indicates whether asynchronous processing is supported.
     *
     * @return {@code false} as asynchronous processing is not supported.
     */
    @Override
    public boolean asyncSupported() {
        return false;
    }

    /**
     * Resets the iterator to its initial state, allowing the data to be iterated over again.
     * In this implementation, it resets the {@code hasNext} flag to {@code true}.
     */
    @Override
    public void reset() {
        hasNext = true;
    }

    /**
     * Returns the size of the current batch of data.
     *
     * @return The number of examples in the dataset.
     */
    @Override
    public int batch() {
        return testingData.numExamples();
    }

    /**
     * Sets a preprocessor to be applied to the dataset before returning it.
     * Note: This implementation does not use a preprocessor.
     *
     * @param preProcessor The {@link org.nd4j.linalg.dataset.api.DataSetPreProcessor} to be set.
     */
    @Override
    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
        // No-op in this implementation
    }

    /**
     * Gets the currently set preprocessor.
     *
     * @return {@code null} as no preprocessor is used in this implementation.
     */
    @Override
    public org.nd4j.linalg.dataset.api.DataSetPreProcessor getPreProcessor() {
        return null;
    }

    /**
     * Returns the labels associated with the dataset.
     *
     * @return A list of labels corresponding to the output classes.
     */
    @Override
    public List<String> getLabels() {
        return Arrays.stream(SignClassification.values()) // Enum representing possible classifications
                .map(SignClassification::name)
                .collect(Collectors.toList());
    }
}
