package org.text2mc;

public class Main {
    public static void main(String[] args) {
        String schemFilePath = args[0];
        String outputDirectoryPath = args[1];
        String outputFileExtension = ".json";
        FileHandler fileHandler = new FileHandler(schemFilePath, outputDirectoryPath, outputFileExtension);
        fileHandler.exportSchematicFile();
    }
}
