package org.text2mc;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.sandrohc.schematic4j.schematic.Schematic;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.io.IOException;

public class FileHandler {
    private String schemFilePath;
    private String outputDirectoryPath;
    private String outputFileExtension;
    private File schemFile;

    public FileHandler(String schemFilePath, String outputDirectoryPath, String outputFileExtension) {
        this.schemFilePath = schemFilePath;
        this.outputDirectoryPath = outputDirectoryPath;
        this.outputFileExtension = outputFileExtension;
        this.schemFile = new File(schemFilePath);
    }

    public String getOutputFilePath(String schemFileName) {
        return outputDirectoryPath + schemFileName + outputFileExtension;
    }

    private boolean fileExists(String filePath) {
        File file = new File(filePath);
        return file.exists();
    }

    public void exportSchematicFile() {
        if (!schemFile.isFile()) {
            System.out.println("Could not find schematic file");
            return;
        }

        String schemFileName = FilenameUtils.removeExtension(schemFile.getName());
        String outputFilePath = getOutputFilePath(schemFileName);

        if (fileExists(outputFilePath)) {
            System.out.println("Schematic output file already exists");
            return;
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
