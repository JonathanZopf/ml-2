package org.example.image_loader;

import org.example.SignClassification;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public record LoadableImage(String path, SignClassification classification) implements Comparable<LoadableImage> {

    public Mat loadMaterial() {
        try {
            Mat image = Imgcodecs.imread(path);
            if (image.empty()) {
                return Mat.zeros(1, 1,  CvType.CV_8UC4);
            }
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGBA);
            return image;
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("Could not load image from path: " + path, e);
        }
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
