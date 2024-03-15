package org.text2mc;

public class World {
    private String[][][] worldArr;
    private int width;
    private int height;
    private int length;

    public World(int width, int height, int length) {
        worldArr = new String[width][height][length];
        this.width = width;
        this.height = height;
        this.length = length;
    }

    public String[][][] getWorldArr() {
        return worldArr;
    }

    public boolean isValidCoordinate(int x, int y, int z) {
        return x >= 0 && x < width &&
                y >= 0 && y < height &&
                z >= 0 && z < length;
    }

    public void initializeCoordinate(String name, int x, int y, int z) {
        if (!isValidCoordinate(x, y, z)) {
            return;
        }
        worldArr[x][y][z] = name;
    }

    public void printWorld() {
        for (int x = 0; x < worldArr.length; x++) {
            for (int y = 0; y < worldArr[x].length; y++) {
                for (int z = 0; z < worldArr[x][y].length; z++) {
                    System.out.print(worldArr[x][y][z] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }
    }
}
