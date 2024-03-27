from world2vec import World2Vec

build_chunks, superflat, superflat_surface = World2Vec.get_build_chunks("mansion/region")

World2Vec.extract_build(build_chunks, superflat, superflat_surface, 1)

World2Vec.export_json_to_npy("testjsons/m35mako.json", "testoutputs/m35mako.npy")