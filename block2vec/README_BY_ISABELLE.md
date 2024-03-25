Setup for World-GAN:

1. Set up conda env + activate
2. Downgrade to Python 3.10 w/ conda
3. Remove all requirement versions from requirements.txt
4. Install reqs from file with pip, also do pip install wandb
5. git init/add submodules for PyAnvilEditor
6. in config.py, swap existing init function w/ this:

   def **init**(self,
   underscores_to_dashes: bool = False,
   explicit_bool: bool = False,
   \*args,
   \*\*kwargs,):
   super().**init**(underscores_to_dashes, explicit_bool, args, kwargs)

7. save and ctrl + click on the super.**init**(..) in the code to open tap.py
8. in tap.py, swap init function head to this:

def **init**(
self,
underscores_to_dashes: bool = False,
explicit_bool: bool = False,
config_files: Optional[List[PathLike]] = None,
\*args,
\*\*kwargs,
): 9. in config.py, change the line with variable "input_dir" to this: input_dir: str = "input\minecraft/Empty_World"

---

Notes:

---

- Neither of the startup commands in the README work :)
- Logic for finding the name of a region file to read based on chunk_pos (chunk position) does not work (see world.py > \_get_region_file). This could be due to using an older version of PyAnvilEditor (if there is one)
- Even if the region file has correct name, zlib cannot read the compression format of the .mca file (using gzip doesn't work either)
