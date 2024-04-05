import multiprocessing
import click

from draw.pipeline import start_continuous_prediction


@click.command(help="Starts Continuous Prediction Pipeline")
def cli_start_pipeline():
    multiprocessing.freeze_support()
    start_continuous_prediction()
