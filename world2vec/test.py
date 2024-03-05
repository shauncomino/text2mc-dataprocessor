from world2vec import World2Vec

build_chunks = World2Vec.get_build_chunks("vex4-large-house-stone-pond/region")

World2Vec.extract_build(build_chunks, 1)
