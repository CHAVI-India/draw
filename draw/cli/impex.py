import click

from draw.config import ALL_SEG_MAP
from draw.impex.export import export_to_zip


@click.command(help="Convert model into ZIP archive")
@click.option(
    "--dataset-id",
    type=int,
    help="3-digit ID for the dataset",
    required=True,
)
@click.option(
    "--model-name",
    type=click.Choice(ALL_SEG_MAP.keys()),
    help="Model name",
    required=True,
)
def cli_export(dataset_id, model_name):
    export_to_zip(dataset_id, model_name)
