import os
import importlib

TEMPLATES = {}  # keys are the prefix of the dataset: ie 'AZ', 'MRA'

data_path = "src/data/sentence_templates"

for filename in os.listdir(data_path):

    if filename.endswith(".py") and filename != "__init__.py":

        modulepath = os.path.join(data_path, filename)
        modulename = filename[:-3]

        spec = importlib.util.spec_from_file_location(modulename, modulepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        key = modulename.split("_")[0]

        TEMPLATES[key] = getattr(module, "AUGMENTED_FEATURE_MEANINGS")
