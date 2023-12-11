import yaml

from schema import Schema, And, SchemaError, Or


def check_schema(conf_schema, conf):
    try:
        conf_schema.validate(conf)
        return True
    except SchemaError as e:
        print(e)
        return False


def get_dict_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


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


PATH = "draw.map/ts_gyne.yml"
print(check_schema(CONF_SCHEMA, get_dict_from_yaml(yaml_path=PATH)))
