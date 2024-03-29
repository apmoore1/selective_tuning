import json
import shutil
import sys
import tempfile

from allennlp.commands import main

#config_file = "./model_configs/bert.jsonnet"
#config_file = "./model_configs/word_embedding.jsonnet"
config_file = "./model_configs/word_embedding_param_groups.jsonnet"
config_file = "./model_configs/bert_layers.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": 0}})

with tempfile.TemporaryDirectory() as serialization_dir:

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", serialization_dir,
        "--include-package", "selective_tuning"
    ]

    main()