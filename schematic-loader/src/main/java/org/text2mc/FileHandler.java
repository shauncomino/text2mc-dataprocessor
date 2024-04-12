package org.text2mc;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.sandrohc.schematic4j.schematic.Schematic;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;

public class FileHandler {
    private String schemFilePath;
    private String outputFilePath;
    private File schemFile;

    public FileHandler(String schemFilePath, String outputFilePath) {
        this.schemFilePath = schemFilePath;
        this.outputFilePath = outputFilePath;
        this.schemFile = new File(schemFilePath);
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
