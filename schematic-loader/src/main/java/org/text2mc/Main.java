package org.text2mc;

public class Main {
    public static void main(String[] args) {
        String schematicDirectoryPath = args[0];
        String outputDirectoryPath = args[1];
        String outputFileExtension = ".json";
        FileHandler fileHandler = new FileHandler(schematicDirectoryPath, outputDirectoryPath, outputFileExtension);
        fileHandler.exportSchematicFiles();
    }
}
