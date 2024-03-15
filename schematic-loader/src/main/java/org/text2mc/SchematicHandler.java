package org.text2mc;

import net.sandrohc.schematic4j.exception.ParsingException;
import net.sandrohc.schematic4j.schematic.Schematic;
import net.sandrohc.schematic4j.schematic.types.*;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class SchematicHandler {
    private Schematic schematic;
    private List<Pair<SchematicBlockPos, SchematicBlock>> blocks;
    private List<SchematicBlockEntity> blockEntities;
    private List<SchematicEntity> entities;

    public SchematicHandler(String schemFilePath) {
        try {
            schematic = net.sandrohc.schematic4j.SchematicLoader.load(schemFilePath);
            blocks = schematic.blocks().collect(Collectors.toList());
            blockEntities = schematic.blockEntities().collect(Collectors.toList());
            entities = schematic.entities().collect(Collectors.toList());
        } catch (ParsingException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Schematic getSchematic() {
        return schematic;
    }

    public List<Pair<SchematicBlockPos, SchematicBlock>> getBlocks() {
        return blocks;
    }

    public List<SchematicBlockEntity> getBlockEntities() {
        return blockEntities;
    }

    public List<SchematicEntity> getEntities() {
        return entities;
    }
}
