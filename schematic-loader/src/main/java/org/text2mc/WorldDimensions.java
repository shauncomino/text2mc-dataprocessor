package org.text2mc;

public class WorldDimensions {
    private int width;
    private int height;
    private int length;

    public WorldDimensions(int width, int height, int length) {
        this.width = width;
        this.height = height;
        this.length = length;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getLength() {
        return length;
    }
}
