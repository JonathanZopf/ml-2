package org.example.image_loader;


import java.util.List;

public record ImageLoaderResult(List<LoadableImage> imagesForTraining, List<LoadableImage> imagesForTesting) {
}
