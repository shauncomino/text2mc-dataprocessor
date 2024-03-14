package org.text2mc;

public class Coordinate {
    private int x;
    private int y;
    private int z;
    private String blockName;

    public Coordinate(int x, int y, int z, String blockName) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.blockName = blockName;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getZ() {
        return z;
    }

    public String getBlockName() {
        return blockName;
    }

    @Override
    public String toString() {
        return "Coordinate{" +
                "x=" + x +
                ", y=" + y +
                ", z=" + z +
                ", blockName='" + blockName + '\'' +
                '}';
    }
}
