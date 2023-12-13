import glob
import os.path

import yaml
from schema import Schema, And, SchemaError, Or

model_schema = Schema(
    {
        "name": str,
        "config": str,
        "map": {And(int, lambda n: n > 0): str},
        "trainer_name": str,
        "postprocess": Or(str, None),
    }
)

CONF_SCHEMA = Schema(
    {
        "name": str,
        "protocol": str,
        "models": {And(int, lambda n: n > 10): model_schema},
    }
)


def get_dict_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def check_yaml_dict_schema(template_schema, read_schema):
    try:
        template_schema.validate(read_schema)
        return read_schema
    except SchemaError as e:
        print(e)
        return None


def get_model_maps(config_root_dir: str) -> tuple[dict, dict]:
    all_seg_map, protocol_to_model = {}, {}
    for file_name in glob.glob(
        os.path.join(config_root_dir, "**", "*.yml"), recursive=True
    ):
        print("Reading", file_name)
        schema = check_yaml_dict_schema(CONF_SCHEMA, get_dict_from_yaml(file_name))
        if schema is not None:
            all_seg_map[schema["name"]] = schema["models"]
            protocol_to_model[schema["protocol"]] = schema["name"]
        else:
            print(file_name, "is erroneous. Skipping...")

    return all_seg_map, protocol_to_model
