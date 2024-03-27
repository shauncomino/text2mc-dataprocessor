package org.text2mc;

import java.util.ArrayList;
import java.util.List;

public class World {
    private List<Block> blocks;
    private WorldDimensions worldDimensions;

    public World(int width, int height, int length) {
        blocks = new ArrayList<>();
        worldDimensions = new WorldDimensions(width, height, length);
    }

    public List<Block> getBlocks() {
        return blocks;
    }

    public WorldDimensions getWorldDimensions() {
        return worldDimensions;
    }

    public boolean isValidCoordinate(int x, int y, int z) {
        return x >= 0 && x < worldDimensions.getWidth() &&
                y >= 0 && y < worldDimensions.getHeight() &&
                z >= 0 && z < worldDimensions.getLength();
    }

    public void initializeCoordinate(String name, int x, int y, int z) {
        if (!isValidCoordinate(x, y, z)) {
            return;
        }
        blocks.add(new Block(name, x, y, z));
    }
}
