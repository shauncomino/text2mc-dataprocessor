#!/bin/python3
import sys
import os
import numpy as np
import pickle
from .pyanvil import World, BlockState, Material


def save_obj(obj, name, prepath='obj/'):
    with open(prepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, prepath='obj/'):
    with open(prepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)

recalc = True
if recalc:
    # with World('Drehmal v2.1 PRIMORDIAL', save_location='C:/Users/awiszus/AppData/Roaming/.minecraft/saves/', debug=True) as wrld:
    with World('Drehmal v2.1 PRIMORDIAL', save_location='/home/awiszus/Project/minecraft_worlds/', debug=True) as wrld:
        regions = os.listdir(wrld.world_folder / 'region')
        blockcounts = dict()
        n_chunks = 0
        for reg_num, r in enumerate(regions):
            name = r.split(".")
            rx = int(name[1])
            rz = int(name[2])
            print("Reading region {}/{} ({}, {})...".format(reg_num, len(regions), rx, rz))
            with open(wrld.world_folder / 'region' / r, mode='rb') as region:
                locations = [[
                    int.from_bytes(region.read(3), byteorder='big', signed=False) * 4096,
                    int.from_bytes(region.read(1), byteorder='big', signed=False) * 4096
                ] for i in range(1024)]
                timestamps = region.read(4096)

                for i, loc in enumerate(locations):
                    try:
                        index = loc[0]
                        if index:
                            cx = index & 0x1f
                            cz = index >> 5

                            cx += rx << 5
                            cz += rz << 5
                            chunk = wrld._load_binary_chunk_at(region, index, loc[1])
                            count = True

                            # name_array = np.empty((16, 256, 16), dtype='object')
                            name_list = []
                            for j in range(0, 16):
                                for k in range(0, 256):
                                    for l in range(0, 16):
                                        try:
                                            block = chunk.get_block((j, k, l))
                                            # name_array[j, k, l] = block.get_state().name
                                            if block.get_state().name not in name_list:
                                                # blockcounts[block.get_state().name] += 1
                                                name_list.append(block.get_state().name)
                                            # else:
                                            #     blockcounts[block.get_state().name] = 1
                                        except IndexError:
                                            # print("Out of scope block")
                                            count = False
                                            break
                            if count:
                                n_chunks += 1
                                for n in name_list:
                                    if n not in blockcounts:
                                        blockcounts[n] = 1
                                    else:
                                        blockcounts[n] += 1

                            print("Successfully read Chunk {}".format(np.unravel_index(i, (32, 32))))
                    except KeyError:
                        print("Cannot find Chunk {}".format(np.unravel_index(i, (32, 32))))
                    except IndexError:
                        print("Cannot index Chunk {}".format(np.unravel_index(i, (32, 32))))
        print('Number of chunks: {}'.format(n_chunks))
        for n in blockcounts.keys():
            blockcounts[n] = blockcounts[n]/n_chunks
        save_obj(blockcounts, 'primordial_counts', prepath='/home/awiszus/Project/minecraft_worlds/results/')
        print('Saved!')
else:
    blockcounts = load_obj('primordial_counts', prepath='/home/awiszus/Project/minecraft_worlds/results/')
    # ch = wrld.get_chunk((1, 1))
    # print('Test!')
    # cv = wrld.get_canvas()
    # cv.select_rectangle((334, 67, -240), (347, 100, -222)).copy().paste(wrld, (411, 105, -302))
    # cv.select_rectangle((334, 67, -240), (347, 100, -222)).fill(BlockState(Material.diamond_block, {}))

    print('Saved!')
