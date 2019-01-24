import sys

import click
from tqdm import tqdm

from src.linbang import LogisticBang
from src.parser import lines_transformer


def print_header(bit_precision, sigma, quadratic_interactions):
    out = "Num weight bits = {}\n".format(bit_precision)
    sys.stdout.write(out)

    out = "Initial sigma = {}\n".format(sigma)
    sys.stdout.write(out)

    if quadratic_interactions:
        out = "Quadratic interactions = {}\n".format(', '.format(quadratic_interactions))
        sys.stdout.write(out)

    out = "average" + "\t\t" + "example" + "\t\t" + "example" + "\t\t" + "current" + "\t\t" + "current" + "\t\t"
    out += "current" + "\n"
    out += "loss" + "\t\t" + "counter" + "\t\t" + "weight" + "\t\t" + "label" + "\t\t" + "predict" + "\t\t"
    out += "features" + "\n"
    sys.stdout.write(out)


@click.command()
@click.option('-q', '--quadratic', 'quadratic_interactions', default='', multiple=True, type=str)
@click.option('-b', '--bit_precision', 'bit_precision', default=23, type=int)
@click.option('-t', '--testonly', 'testonly', default=False, type=bool)
@click.option('-s', '--sigma', 'sigma', default=0.01, type=float)
@click.option('-m', '--predict_mode', 'predict_mode', default=True, type=bool)
@click.option('--progress', 'progress', default=100, type=int)
def bang(quadratic_interactions, bit_precision, testonly, predict_mode, sigma, progress):
    print_header(bit_precision, sigma, quadratic_interactions)

    model = LogisticBang(bit_precision, sigma)

    rows = click.get_text_stream('stdin')
    j = progress
    for i, row in enumerate(lines_transformer(rows, quadratic_interactions, bit_precision)):
        if not testonly:
            model.partial_fit(row)
        if predict_mode:
            prediction = model.predict(row)
        else:
            prediction = model.sample_predict(row)

        label, weight, _, features = row
        if not j:
            out = "%.4f" % model.average_loss + "\t\t" + str(model.example_counter) + "\t\t" + str(weight) + "\t\t"
            out += str(label) + "\t\t" + "%.4f" % prediction + "\t\t" + str(len(features)) + "\n"
            sys.stdout.write(out)
            j = progress
        j -= 1


if __name__ == "__main__":
    bang()
