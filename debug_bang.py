import click
from tqdm import tqdm
import sys

from src.linbang import LogisticBang
from src.parser import hasher, lines_transformer


def validate_progress(ctx, param, value):
    try:
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except:
            raise ValueError
    except ValueError:
        raise click.BadParameter(
            'Error: Invalid value for "-P" / "--progress": {} is not a valid integer/float'.format(value))



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
@click.option('-b', '--bit_precision', 'bit_precision', default=9, type=int)
@click.option('-t', '--testonly/--learn', 'testonly', default=False)
@click.option('--l2', 'reg', default=2, type=float)
@click.option('-m', '--mode/--sampling', 'mode', default=True)
@click.option('-f', '--final_regressor', 'final_regressor', default='', type=click.Path(exists=False))
@click.option('-i', '--initial_regressor', 'initial_regressor', default='', type=click.Path(exists=False))
@click.option('-P', '--progress', 'progress', callback=validate_progress, default="1000")
@click.option('--quiet/--no-quiet', 'quiet', default=False)
def bang(quadratic_interactions, bit_precision, testonly, mode, reg, progress, final_regressor,
         initial_regressor, quiet):
    print_header(bit_precision, reg, quadratic_interactions)
    model = LogisticBang(bit_precision, reg)

    with open("./data/datasets/rcv1.train.vw") as f:
        rows = f.readlines()  # [:500000]

    rows_iterator = lines_transformer(rows, quadratic_interactions, bit_precision)
    for i, row in enumerate(rows_iterator):
        if mode:
            prediction = model.predict(row)
        else:
            prediction = model.sample_predict(row)

        if not testonly:
            model.partial_fit(row)

        label, weight, _, features = row
        if not i % progress:
            if quiet:
                out = "%.4f" % prediction + "\n"
                sys.stdout.write(out)
            else:
                out = "%.4f" % model.average_loss + "\t\t" + str(model.example_counter) + "\t\t" + str(weight) + "\t\t"
                out += str(label) + "\t\t" + "%.4f" % prediction + "\t\t" + str(len(features)) + "\n"
                sys.stdout.write(out)
            if isinstance(progress, float):
                progress *= progress

    model

if __name__ == "__main__":
    import pickle

    # with open("./data/models/model1.pkl", "rb") as f:
    #     d = pickle.load(f)
    # d
    bang()
