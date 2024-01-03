import click

from draw.pipeline import start_continuous_prediction


@click.command(help="Starts Continuous Prediction Pipeline")
def cli_start_pipeline():
    start_continuous_prediction()
