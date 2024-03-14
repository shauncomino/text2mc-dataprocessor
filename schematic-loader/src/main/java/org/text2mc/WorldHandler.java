package org.text2mc;

import net.sandrohc.schematic4j.schematic.types.Pair;
import net.sandrohc.schematic4j.schematic.types.SchematicBlock;
import net.sandrohc.schematic4j.schematic.types.SchematicBlockPos;

import java.util.List;

public class WorldHandler {
    private World world;

    public WorldHandler(World world) {
        this.world = world;
    }

    public void addBlocks(List<Pair<SchematicBlockPos, SchematicBlock>> blocks) {
        for(Pair<SchematicBlockPos, SchematicBlock> block : blocks) {
            int x = block.left.x;
            int y = block.left.y;
            int z = block.left.z;
            String blockName = block.right.name;
            world.initializeCoordinate(blockName, x, y, z);
        }
    }
}
