package org.text2mc;

public class Block {
    private String name;
    private int x;
    private int y;
    private int z;

    public Block(String name, int x, int y, int z) {
        this.name = name;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public String getName() {
        return name;
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

    @Override
    public String toString() {
        return "{name=" + name + ", x=" + x + ", y=" + y + ", z=" + z + "}";
    }
}
