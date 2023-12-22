import click

from a9t.cli import (
    cli_prepare_and_train,
    cli_predict,
    cli_preprocess,
    cli_start_pipeline,
    cli_export,
)


@click.group(
    name="a9t",
    help="AutoSegmentation Pipeline based on NNUNet",
)
def cli():
    pass


cli.add_command(cli_prepare_and_train, "train-single-gpu")
cli.add_command(cli_predict, "predict")
cli.add_command(cli_preprocess, "preprocess")
cli.add_command(cli_start_pipeline, "start-pipeline")
cli.add_command(cli_export, "zip-model")


if __name__ == "__main__":
    cli()
