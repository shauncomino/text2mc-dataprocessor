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

            int width = schematic.width();
            int height = schematic.height();
            int length = schematic.length();

            String[][][] blockNames = new String[width][height][length];

            for (Pair<SchematicBlockPos, SchematicBlock> block : blocks) {
                int x = block.left.x;
                int y = block.left.y;
                int z = block.left.z;
                String blockName = block.right.name;
                blockNames[x][y][z] = blockName;
            }

            printBlockNames(blockNames);
        } catch (ParsingException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void printBlockNames(String[][][] blockNames) {
        for (int i = 0; i < blockNames.length; i++) {
            for (int j = 0; j < blockNames[i].length; j++) {
                for (int k = 0; k < blockNames[i][j].length; k++) {
                    System.out.print("(" + i + "," + j + "," + k + "): " + blockNames[i][j][k] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }
    }
}
