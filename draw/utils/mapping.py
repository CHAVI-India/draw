import glob
import os.path

import yaml
from schema import Schema, And, SchemaError, Or

from draw.utils.logging import get_log

LOG = get_log()
ENV_FILE_NAME = "env.draw.yml"


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
        # 0-10 reserved for MSD. Avoid
        "models": {And(int, lambda n: n > 10): model_schema},
    }
)

ENV_SCHEMA = Schema(
    {
        "DB_URL": str,
        "DB_NAME": str,
        "TABLE_NAME": str,
        "WATCH_DIR": str,
        "MODEL_DEF_ROOT": str,
    }
)


def get_dict_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError:
            LOG.warning(f"Exception while reading YAML {yaml_path} %s", exc_info=True)


def check_yaml_dict_schema(template_schema, read_schema):
    try:
        template_schema.validate(read_schema)
        return read_schema
    except SchemaError:
        LOG.error(f"Error while reading schema", exc_info=True)
        return None


def get_model_maps(config_root_dir: str) -> tuple[dict, dict]:
    all_seg_map, protocol_to_model = {}, {}
    for file_name in glob.glob(
        os.path.join(config_root_dir, "**", "*.yml"), recursive=True
    ):
        schema = check_yaml_dict_schema(CONF_SCHEMA, get_dict_from_yaml(file_name))
        if schema is not None:
            all_seg_map[schema["name"]] = schema["models"]
            protocol_to_model[schema["protocol"]] = schema["name"]
        else:
            LOG.warning(f"Skipped {file_name} due to schema problems")

    return all_seg_map, protocol_to_model


def get_env_map():
    return check_yaml_dict_schema(ENV_SCHEMA, get_dict_from_yaml(ENV_FILE_NAME))
