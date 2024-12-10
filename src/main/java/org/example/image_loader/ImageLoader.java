package org.example.image_loader;

import org.example.SignClassification;

import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class ImageLoader {
    private final int imagesForTraining;
    private final int imagesForTesting;

    public ImageLoader(int imagesForTraining, int imagesForTesting) {
        this.imagesForTraining = imagesForTraining;
        this.imagesForTesting = imagesForTesting;
    }

    public ImageLoaderResult loadImages() {
        List<LoadableImage> availableSigns = getAllSigns();
        if (availableSigns.size() < imagesForTraining + imagesForTesting) {
            throw new IllegalArgumentException("Not enough images for training and testing");
        }

        availableSigns.sort(LoadableImage::compareTo);
        List<LoadableImage> imagesForTraining =  availableSigns.subList(0, this.imagesForTraining);
        List<LoadableImage> imagesForTesting = availableSigns.subList(availableSigns.size() - this.imagesForTraining - 1, availableSigns.size() - 1);

        return new ImageLoaderResult(imagesForTraining, imagesForTesting);
     }

    private static List<LoadableImage> getAllSigns() {
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        InputStream fileLocationStream = classloader.getResourceAsStream("file_paths/images.txt");

        if (fileLocationStream == null) {
            throw new IllegalArgumentException("There is no file_paths/images.txt in the resources folder");
        }

        String parentFolderLocation;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(fileLocationStream))) {
            parentFolderLocation = reader.lines().collect(Collectors.joining("\n"));
        } catch (IOException e) {
            throw new RuntimeException("Error reading file_location.txt", e);
        }

        File parentDir = new File(parentFolderLocation);
        if (!parentDir.exists() || !parentDir.isDirectory()) {
            throw new IllegalArgumentException("There is no parent directory in " + parentDir.getAbsolutePath());
        }

        File[] childDirs = parentDir.listFiles(File::isDirectory);
        if (childDirs == null || childDirs.length == 0) {
            throw new IllegalArgumentException("There are no child directories in " + parentDir.getAbsolutePath());
        }

        List<LoadableImage> allSigns = new ArrayList<>();
        for (File dir : childDirs) {
            List<String> filePaths = getAllLoadableFilesInDir(dir);
            allSigns.addAll(filePaths.stream()
                    .map(path -> new LoadableImage(path, SignClassification.valueOf(dir.getName())))
                    .toList());
        }

        return allSigns;
    }

    private static List<String> getAllLoadableFilesInDir(File directory) {
        if (!directory.exists() || !directory.isDirectory()) {
            throw new IllegalArgumentException("Invalid directory: " + directory.getAbsolutePath());
        }

        try {
            return Files.walk(directory.toPath())
                    .filter(Files::isRegularFile)
                    .map(path -> path.toAbsolutePath().toString())
                    .filter(path -> path.endsWith(".jpg") || path.endsWith(".bmp"))
                    .collect(Collectors.toList());
        } catch (IOException e) {
            throw new RuntimeException("Error reading files from directory: " + directory.getAbsolutePath(), e);
        }
    }

}
