package org.example;

import org.example.image_loader.ImageLoader;
import org.example.image_loader.ImageLoaderResult;
import org.example.image_loader.LoadableImage;

public class Main {
    public static void main(String[] args) {
        int imagesForTraining = 100;
        int imagesForTesting = 50;

        ImageLoader loader = new ImageLoader(imagesForTraining, imagesForTesting);
        ImageLoaderResult loaderResult = loader.loadImages();
        System.out.println("Images for training:");
        for(LoadableImage image : loaderResult.imagesForTraining()) {
            System.out.println(image.path());
        }

        System.out.println("Images for testing:");
        for(LoadableImage image : loaderResult.imagesForTesting()) {
            System.out.println(image.path());
        }
    }
}