package org.text2mc;

import net.sandrohc.schematic4j.schematic.Schematic;

public class Main {
    public static void main(String[] args) {
        String schemFilePath = "src/main/resources/12727.schematic";

        SchematicHandler schematicHandler = new SchematicHandler(schemFilePath);

        Schematic schematic = schematicHandler.getSchematic();
        int width = schematic.width();
        int height = schematic.height();
        int length = schematic.length();

        World world = new World(width, height, length);
        WorldHandler worldHandler = new WorldHandler(world);
        worldHandler.addBlocks(schematicHandler.getBlocks());

        world.printWorld();
    }
}
