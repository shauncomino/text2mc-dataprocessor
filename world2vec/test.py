from world2vec import World2Vec

build_chunks, superflat = World2Vec.get_build_chunks("bluecity/region")

World2Vec.extract_build(build_chunks, superflat, 123)