import click
from tqdm import tqdm
import sys

from src.linbang import LogisticBang
from src.parser import hasher, lines_transformer

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
def bang(quadratic_interactions, bit_precision, testonly, predict_mode, sigma):
    print_header(bit_precision, sigma, quadratic_interactions)
    model = LogisticBang(bit_precision, sigma)

    with open("./data/rcv1.test.vw") as f:
        rows = f.readlines()[:5000]

    rows_iterator = lines_transformer(rows, quadratic_interactions, bit_precision)
    for row in rows_iterator:
        model.partial_fit(row)
    model.predict(row)


    # rows = [
    #     "1 1.0 zebra|MetricFeatures:3.28 height:1.5 length:2.0|etricFeatures2:3.28 height2:1.5 length2:2.0 width2:1|tricFeatures2:3.28 height2:1.5 length2:2.0 width2:1",
    #     "0 1.0 zebra|MetricFeatures:3.28 height:1.5 length:2.0|etricFeatures2:3.28 height2:1.5 length2:2.0 width2:1|tricFeatures2:3.28 height2:1.5 length2:2.0 width2:1"]
    # rows *= 200
    # rows_iterator = lines_transformer(rows, quadratic_interactions)
    #
    # prds = []
    # for row in tqdm(rows_iterator):
    #     prd = model.sample_predict(row)
    #     prds.append(prd)
    # mean = sum(prds) / len(prds)
    # mean


if __name__ == "__main__":
    bang()
