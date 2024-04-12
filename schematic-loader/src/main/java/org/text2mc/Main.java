package org.text2mc;

public class Main {
    public static void main(String[] args) {
        String schemFilePath = args[0];
        String outputFilePath = args[1];
        FileHandler fileHandler = new FileHandler(schemFilePath, outputFilePath);
        fileHandler.exportSchematicFile();
    }
}
