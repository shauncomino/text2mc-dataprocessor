from world2vec import World2Vec

build_chunks, superflat, superflat_surface = World2Vec.get_build_chunks("mansion/region")

World2Vec.extract_build(build_chunks, superflat, superflat_surface, 1)

world_arr = World2Vec.export_json_to_npy("testjsons/m35mako.json")
World2Vec.export_npy_to_hdf5("m35mako", world_arr)