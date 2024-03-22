package org.text2mc;

public class Main {
    public static void main(String[] args) {
        String schematicDirectoryPath = "src/main/resources/";
        String outputDirectoryPath = "src/main/output/";
        String outputFileExtension = ".out";
        FileHandler fileHandler = new FileHandler(schematicDirectoryPath, outputDirectoryPath, outputFileExtension);
        fileHandler.exportSchematicFiles();
    }
}
