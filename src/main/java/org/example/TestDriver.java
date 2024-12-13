package org.example;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.example.deep_learing_network.Evaluator;
import org.example.deep_learing_network.ModelBuilder;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

import static org.example.Main.SCALE_TARGET_PIXEL_SIZE_COLS;
import static org.example.Main.SCALE_TARGET_PIXEL_SIZE_ROWS;

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

private double runForFunction(Activation activationFunction, DataSet trainingData, DataSet testingData) {
        MultiLayerNetwork model = new ModelBuilder()
                .withInputSize(SCALE_TARGET_PIXEL_SIZE_ROWS * SCALE_TARGET_PIXEL_SIZE_COLS * 4)
                .withOutputSize(SignClassification.values().length)
                .withLearningRate(0.001)
                .withHiddenLayerSize(128)
                .withHiddenLayerActivation(Activation.RELU)
                .withOutputLayerActivation(Activation.SOFTMAX)
                .withNumEpochs(100)
                .withLogFrequency(10)
                .buildAndTrain(trainingData);

    Evaluator evaluator = new Evaluator(model, testingData);
    Evaluation result = evaluator.getEvaluationResult();
    return result.accuracy();
    }

void testFunctionsAccuracy() {
    List<List<Activation>> groupedActivations = List.of(
            // 1. Lineare und Identitätsfunktionen
            List.of(Activation.IDENTITY),

            // 2. Exponentielle und Logistische Funktionen
            List.of(Activation.SIGMOID, Activation.HARDSIGMOID, Activation.SOFTMAX, Activation.SOFTSIGN),

            // 3. Hyperbolische Tangens-Funktionen
            List.of(Activation.TANH, Activation.HARDTANH, Activation.RATIONALTANH, Activation.RECTIFIEDTANH),

            // 4. ReLU-Varianten
            List.of(Activation.RELU, Activation.LEAKYRELU, Activation.RELU6, Activation.RRELU,
                    Activation.THRESHOLDEDRELU, Activation.ELU, Activation.SELU, Activation.GELU),

            // 5. Nichtlineare polynomiale Funktionen
            List.of(Activation.CUBE),

            // 6. Glatte Funktionen für Optimierungen
            List.of(Activation.SOFTPLUS, Activation.SWISH, Activation.MISH)
    );
}

}
