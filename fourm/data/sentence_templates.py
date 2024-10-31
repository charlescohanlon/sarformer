import os
import importlib

from Augmented_Text_Dicts import *

print(os.getcwd())

all_sentence_templates = {}
data_path = 'fourm\data\Augmented_Text_Dicts'

for filename in os.listdir(data_path):


    if filename.endswith(".py") and filename != "__init__.py":
        
        modulepath = os.path.join(data_path, filename)
        modulename = filename[:-3]

        spec = importlib.util.spec_from_file_location(modulename, modulepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        key = modulename # TODO change later based on what the uids look like

        all_sentence_templates[key] = getattr(module, modulename)

