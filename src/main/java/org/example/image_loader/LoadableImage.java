package org.example.image_loader;

import org.example.SignClassification;
import org.opencv.core.Mat;
public record LoadableImage(String path, SignClassification classification) implements Comparable<LoadableImage> {

    public Mat loadMaterial() {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public int compareTo(LoadableImage o) {
        int aHash = this.hashCode();
        int bHash = o.hashCode();
        if (aHash == bHash) {
            return 0;
        }
        return aHash > bHash ? 1 : -1;
    }
}
