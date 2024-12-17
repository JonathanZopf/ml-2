package org.example;

import kotlin.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestDriver {
/**
 * Ideen für den Vortrag:
 * - Gruppierung und Vergleich verschiedener Aktivierungsfunktionen nach Typen (z. B. linear, nicht-linear,
 *   schwellwertbasiert).
 * - Untersuchung: Hat die Verwendung unterschiedlicher Aktivierungsfunktionen in verschiedenen Layers
 *   Einfluss auf das Ergebnis?
 * - Sinnhaftigkeit der Anpassung des Parameters in der Sigmoid-Funktion analysieren.
 *
 * Allgemein:
 * - Präsentation auf abstrakter Ebene halten, ohne Fokus auf spezifische Frameworks.
 * - Dauer: 20 Minuten, Mittwoch, 9:15 Uhr.
 * - Inhalte: Kurzvorstellung von Merkmalsextraktion und Netzaufbau.
 */

private final TestFunction testFunction;

    private final List<Activation> activationFunctionsForHiddenLayer = List.of(
            // Linear Activations
            Activation.IDENTITY
/*
            // Non-linear Activations
            // Threshold-based
            Activation.RELU,
            Activation.HARDTANH,
            Activation.HARDSIGMOID,
            // Smooth
            Activation.SIGMOID,
            Activation.SELU,
            Activation.SWISH,
            Activation.GELU,
            Activation.MISH,
            Activation.TANH,
            Activation.SOFTPLUS*/
    );


public TestDriver(TestFunction testFunction) {
    this.testFunction = testFunction;
}

public void determineBestHiddenLayersActivationFunction() {
    List<Pair<String, Evaluation>> results = new ArrayList<>();
    for (Activation function : activationFunctionsForHiddenLayer) {
        System.out.println("Testing activation function: " + function.name());
        results.add(new Pair<>(function.name(), testFunction.testDetermineBestHiddenLayersActivationFunction(function)));
    }

    results.sort((a, b) -> Double.compare(b.getSecond().accuracy(), a.getSecond().accuracy()));
    System.out.println("---------------------------------Hidden Layer Activation Function Comparison---------------------------------");
    results.forEach((result) -> {
        System.out.println("Activation Function: " + result.getFirst());
        printEvaluationResult(result.getSecond());
        System.out.println("---------------------------------");
    });
}

public void testEffectivityOfDifferentHiddenLayerActivationFunction() {
        List<Pair<Integer, Activation>> baseConfig = Arrays.asList(
                new Pair<>(500, Activation.RELU),
                new Pair<>(250, Activation.RELU),
                new Pair<>(128, Activation.RELU),
                new Pair<>(64, Activation.RELU)
        );

    List<Pair<Integer, Activation>> modifiedConfig1 = List.of(
            new Pair<>(500, Activation.RELU),
            new Pair<>(250, Activation.RELU),
            new Pair<>(128, Activation.GELU),
            new Pair<>(64, Activation.SWISH)
    );


    List<Pair<Integer, Activation>> modifiedConfig2 = List.of(
            new Pair<>(500, Activation.RELU),
            new Pair<>(250, Activation.RELU),
            new Pair<>(128, Activation.SELU),
            new Pair<>(64, Activation.SELU)
    );

    List<Pair<Integer, Activation>> modifiedConfig3 = List.of(
            new Pair<>(500, Activation.RELU),
            new Pair<>(250, Activation.ELU),
            new Pair<>(128, Activation.ELU),
            new Pair<>(64, Activation.MISH)
    );


    List<Pair<String, List<Pair<Integer, Activation>> >> configsToTest = List.of(
                new Pair<>("Base Configuration", baseConfig),
                new Pair<>("Modified Configuration 1", modifiedConfig1),
                new Pair<>("Modified Configuration 2", modifiedConfig2),
                new Pair<>("Modified Configuration 3", modifiedConfig3)
        );

        var results = configsToTest.stream().map(config -> {
            System.out.println("Testing configuration: " + config.getFirst());
            return new Pair<>(config.getFirst(), testFunction.testDetermineBestHiddenLayersActivationFunction(config.getSecond()));
        }).toList();

        System.out.println("---------------------------------Hidden Layer Activation Function Comparison---------------------------------");
        results.forEach((result) -> {
            System.out.println("Configuration: " + result.getFirst());
            printEvaluationResult(result.getSecond());
            System.out.println("---------------------------------");
        });

    }


    public void testParameterAdjustmentInSigmoidFunction() {
        List<Double> parameters = List.of(0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0);
        List<Pair<Double, Evaluation>> results = new ArrayList<>();
        for (double parameter : parameters) {
            System.out.println("Testing parameter: " + parameter);
            results.add(new Pair<>(parameter, testFunction.testParameterAdjustmentInSigmoidFunction(parameter)));
        }

        System.out.println("---------------------------------Sigmoid Parameter Adjustment---------------------------------");
        results.forEach((result) -> {
            System.out.println("Parameter: " + result.getFirst());
            printEvaluationResult(result.getSecond());
            System.out.println("---------------------------------");
        });
    }

    private void printEvaluationResult(Evaluation evaluation) {
        System.out.println("Accuracy: " + evaluation.accuracy());
        System.out.println("Precision: " + evaluation.precision());
        System.out.println("Recall: " + evaluation.recall());
        System.out.println("F1 Score: " + evaluation.f1());
    }
}