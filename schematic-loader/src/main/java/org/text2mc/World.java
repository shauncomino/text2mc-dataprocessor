package org.text2mc;

public class World {
    private String[][][] worldArr;

    public World(int width, int height, int length) {
        worldArr = new String[width][height][length];
    }

    public String[][][] getWorldArr() {
        return worldArr;
    }

    public void initializeCoordinate(String name, int x, int y, int z) {
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
