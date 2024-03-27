package org.text2mc;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.sandrohc.schematic4j.schematic.Schematic;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.io.IOException;

public class FileHandler {
    private String schemDirectoryPath;
    private String outputDirectoryPath;
    private String outputFileExtension;
    private File directory;
    private File[] schemFiles;

    public FileHandler(String schemDirectoryPath, String outputDirectoryPath, String outputFileExtension) {
        this.schemDirectoryPath = schemDirectoryPath;
        this.outputDirectoryPath = outputDirectoryPath;
        this.outputFileExtension = outputFileExtension;
        this.directory = new File(schemDirectoryPath);
        this.schemFiles = this.directory.listFiles();
    }

    public String getOutputFilePath(String schemFileName) {
        return outputDirectoryPath + schemFileName + outputFileExtension;
    }

    private boolean fileExists(String filePath) {
        File file = new File(filePath);
        return file.exists();
    }

    public void exportSchematicFiles() {
        if (schemFiles == null) {
            System.out.println("No schematic files found");
            return;
        }

        for (File schemFile : schemFiles) {
            String schemFilePath = schemFile.getPath();
            String schemFileName = FilenameUtils.removeExtension(schemFile.getName());
            String outputFilePath = getOutputFilePath(schemFileName);

            if (fileExists(outputFilePath)) {
                continue;
            }

            SchematicHandler schematicHandler = new SchematicHandler(schemFilePath);
            Schematic schematic = schematicHandler.getSchematic();
            int width = schematic.width();
            int height = schematic.height();
            int length = schematic.length();

            World world = new World(width, height, length);
            WorldHandler worldHandler = new WorldHandler(world);
            worldHandler.addBlocks(schematicHandler.getBlocks());
            worldHandler.addBlockEntities(schematicHandler.getBlockEntities());
            worldHandler.addEntities(schematicHandler.getEntities());
            writeSchematicToFile(outputFilePath, world);
        }
    }

    private String getOutputString(World world) {
        Gson gson = new GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create();
        return gson.toJson(world);
    }

    private void writeSchematicToFile(String filePath, World world) {
        try {
            File file = new File(filePath);
            String output = getOutputString(world);
            FileUtils.write(file, output, "UTF-8", true);
        } catch (IOException e) {
            System.out.println(e);
        }
    }
}
