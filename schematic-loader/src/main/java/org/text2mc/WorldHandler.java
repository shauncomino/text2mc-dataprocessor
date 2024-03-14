package org.text2mc;

import net.sandrohc.schematic4j.schematic.types.Pair;
import net.sandrohc.schematic4j.schematic.types.SchematicBlock;
import net.sandrohc.schematic4j.schematic.types.SchematicBlockEntity;
import net.sandrohc.schematic4j.schematic.types.SchematicBlockPos;

import java.util.List;

public class WorldHandler {
    private World world;

    public WorldHandler(World world) {
        this.world = world;
    }

    public void addBlocks(List<Pair<SchematicBlockPos, SchematicBlock>> blocks) {
        for (Pair<SchematicBlockPos, SchematicBlock> block : blocks) {
            int x = block.left.x;
            int y = block.left.y;
            int z = block.left.z;
            String blockName = block.right.name;
            world.initializeCoordinate(blockName, x, y, z);
        }
    }

    public void addBlockEntities(List<SchematicBlockEntity> blockEntities) {
        for (SchematicBlockEntity blockEntity : blockEntities) {
            int x = blockEntity.pos.x;
            int y = blockEntity.pos.y;
            int z = blockEntity.pos.z;
            String blockEntityName = blockEntity.name;
            world.initializeCoordinate(blockEntityName, x, y, z);
        }
    }
}
