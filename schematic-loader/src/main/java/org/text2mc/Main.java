package org.text2mc;

import net.sandrohc.schematic4j.SchematicLoader;
import net.sandrohc.schematic4j.exception.ParsingException;
import net.sandrohc.schematic4j.schematic.Schematic;
import net.sandrohc.schematic4j.schematic.types.*;

import java.io.*;
import java.util.List;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) {
        String schemFilePath = "src/main/resources/12727.schematic";

        try {
            Schematic schematic = SchematicLoader.load(schemFilePath);
            List<Pair<SchematicBlockPos, SchematicBlock>> blocks = schematic.blocks().collect(Collectors.toList());
            List<SchematicBlockEntity> blockEntities = schematic.blockEntities().collect(Collectors.toList());
            List<SchematicEntity> entities = schematic.entities().collect(Collectors.toList());

            int length = schematic.length();
            int width = schematic.width();
            int height = schematic.height();

            System.out.println(length + " " + width + " " + height);

            for (Pair<SchematicBlockPos, SchematicBlock> block : blocks) {
                int x = block.left.x;
                int y = block.left.y;
                int z = block.left.z;
                String blockName = block.right.name;
                Coordinate coord = new Coordinate(x, y, z, blockName);
                System.out.println(coord);
            }
        } catch (ParsingException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}