package org.example;

import kotlin.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;

import java.util.List;

public interface TestFunction {
    Evaluation testDetermineBestHiddenLayersActivationFunction(Activation hiddenLayerActivationFunction);
    Evaluation testDetermineBestHiddenLayersActivationFunction(List<Pair<Integer, Activation>> hiddenLayerConfig);

    Evaluation testParameterAdjustmentInSigmoidFunction(double parameter);

}
