from world2vec import World2Vec

build_chunks = World2Vec.get_build_chunks("test_regions1")

World2Vec.extract_build(build_chunks, "test_build1")

build_chunks = World2Vec.get_build_chunks("test_regions2")

World2Vec.extract_build(build_chunks, "test_build2")