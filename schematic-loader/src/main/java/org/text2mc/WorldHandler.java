package org.text2mc;

import net.sandrohc.schematic4j.schematic.types.*;

import java.util.List;

public class WorldHandler {
    private World world;

    public WorldHandler(World world) {
        this.world = world;
    }

    public void addBlocks(List<Pair<SchematicBlockPos, SchematicBlock>> blocks) {
        try {
            for (Pair<SchematicBlockPos, SchematicBlock> block : blocks) {
                int x = block.left.x;
                int y = block.left.y;
                int z = block.left.z;
                String blockName = block.right.name;
                world.initializeCoordinate(blockName, x, y, z);
            }
        } catch (Exception e) {
            System.out.println(e);
        }
    }

    public void addBlockEntities(List<SchematicBlockEntity> blockEntities) {
        try {
            for (SchematicBlockEntity blockEntity : blockEntities) {
                int x = blockEntity.pos.x;
                int y = blockEntity.pos.y;
                int z = blockEntity.pos.z;
                String blockEntityName = blockEntity.name;
                world.initializeCoordinate(blockEntityName, x, y, z);
            }
        } catch (Exception e) {
            System.out.println(e);
        }
    }

    public void addEntities(List<SchematicEntity> entities) {
        try {
            for (SchematicEntity entity : entities) {
                int x = (int) Math.floor(entity.pos.x);
                int y = (int) Math.floor(entity.pos.y);
                int z = (int) Math.floor(entity.pos.z);
                String entityName = entity.name;
                world.initializeCoordinate(entityName, x, y, z);
            }
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
