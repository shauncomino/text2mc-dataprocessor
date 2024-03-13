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

            System.out.println(schematic.name());
            System.out.println(schematic.width());
            System.out.println(schematic.height());
            System.out.println(schematic.length());
            System.out.println();
            System.out.println(blocks);
            System.out.println();
            System.out.println(blockEntities);
            System.out.println();
            System.out.println(entities);
        } catch (ParsingException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}