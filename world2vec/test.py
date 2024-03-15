from world2vec import World2Vec

build_chunks, superflat, superflat_surface = World2Vec.get_build_chunks("mansion/region")

World2Vec.extract_build(build_chunks, superflat, superflat_surface, 1)