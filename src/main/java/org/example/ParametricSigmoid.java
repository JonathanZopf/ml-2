package org.example;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ParametricSigmoid extends BaseActivationFunction {
    private final double k;

    public ParametricSigmoid(double k) {
        this.k = k;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        // Apply sigmoid function h(x) = 1 / (1 + exp(-k * x))
        INDArray negKX = in.mul(-k); // Compute -k * x
        INDArray expNegKX = Transforms.exp(negKX); // Compute exp(-k * x)
        in.assign(expNegKX.rdiv(1.0).addi(1.0).rdivi(1.0)); // h(x) = 1 / (1 + exp(-k * x))
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        // Compute h(x) = 1 / (1 + exp(-k * x))
        INDArray negKX = in.mul(-k);
        INDArray expNegKX = Transforms.exp(negKX);
        INDArray hX = expNegKX.rdiv(1.0).addi(1.0).rdivi(1.0); // h(x)

        // Compute h'(x) = k * h(x) * (1 - h(x))
        INDArray dLdz = hX.mul(hX.rsub(1.0)).muli(k); // h(x) * (1 - h(x)) * k

        // Multiply by epsilon (chain rule)
        dLdz.muli(epsilon);

        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "ParametricSigmoidActivation(k=" + k + ")";
    }
}
