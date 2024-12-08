package org.example;

import java.awt.*;

public class PixelValues {
   private final double normalizedRed;
    private final double normalizedGreen;
    private final double normalizedBlue;
    private final double normalizedAlpha;

    private PixelValues(double normalizedRed, double normalizedGreen, double normalizedBlue, double normalizedAlpha) {
        this.normalizedRed = normalizedRed;
        this.normalizedGreen = normalizedGreen;
        this.normalizedBlue = normalizedBlue;
        this.normalizedAlpha = normalizedAlpha;
    }

    public double getNormalizedRed() {
        return normalizedRed;
    }

    public double getNormalizedGreen() {
        return normalizedGreen;
    }

    public double getNormalizedBlue() {
        return normalizedBlue;
    }

    public double getNormalizedAlpha() {
        return normalizedAlpha;
    }

    public Color getColor() {
        return new Color(
            (float) normalizedRed,
            (float) normalizedGreen,
            (float) normalizedBlue,
            (float) normalizedAlpha
        );
    }
    public static PixelValues fromUnormalizedArray(double[] pixelArray) {
        return new PixelValues(
            pixelArray[0] / 255.0,
            pixelArray[1] / 255.0,
            pixelArray[2] / 255.0,
            pixelArray[3] / 255.0
        );
    }

}
