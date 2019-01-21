import click
from tqdm import tqdm

from src.linbang import LogisticBang
from src.parser import hasher, lines_transformer


@click.command()
@click.option('-q', '--quadratic', 'quadratic_interactions', default='', multiple=True, type=str)
@click.option('-b', '--bit_precision', 'bit_precision', default=23, type=int)
def bang(quadratic_interactions, bit_precision):
    print(quadratic_interactions, bit_precision)
    feature_range = 2 ** bit_precision

    hasher.range = feature_range
    model = LogisticBang(feature_range, 0.1)

    rows = click.get_text_stream('stdin')
    for row in tqdm(lines_transformer(rows, quadratic_interactions)):
        model.partial_fit(row)
        print(model.example_counter, model.average_loss)

if __name__ == "__main__":
    bang()
