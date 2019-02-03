import sys

import click

from src.linbang import LogisticBang
from src.parser import lines_transformer


def validate_progress(ctx, param, value):
    try:
        if value.isdigit():
            return int(value)
        else:
            return float(value)
    except ValueError:
        raise click.BadParameter(
            'Error: Invalid value for "-P" / "--progress": {} is not a valid integer/float'.format(value))
    except AttributeError:
        return 1


def get_header(bit_precision, reg, quadratic_interactions, keep_namespaces, ignore_namespaces):
    if ignore_namespaces:
        out = "Ignoring namespaces beginning with: {}\n".join(','.join(ignore_namespaces))
        sys.stdout.write(out)

    if keep_namespaces:
        out = "Using namespaces beginning with: {}\n".join(','.join(keep_namespaces))
        sys.stdout.write(out)

    out = "Num weight bits = {}\n".format(bit_precision)
    sys.stdout.write(out)

    out = "Initial reg = {}\n".format(reg)
    sys.stdout.write(out)

    if quadratic_interactions:
        out = "Quadratic interactions = {}\n".format(', '.format(quadratic_interactions))
        sys.stdout.write(out)

    out = "average" + "\t\t" + "example" + "\t\t" + "example" + "\t\t" + "current" + "\t\t" + "current" + "\t\t"
    out += "current" + "\n"
    out += "loss" + "\t\t" + "counter" + "\t\t" + "weight" + "\t\t" + "label" + "\t\t" + "predict" + "\t\t"
    out += "features" + "\n"
    return out


@click.command()
@click.option('-q', '--quadratic', 'quadratic_interactions', default='', multiple=True, type=str)
@click.option('--keep', 'keep_namespaces', default='', multiple=True, type=str)
@click.option('--ignore', 'ignore_namespaces', default='', multiple=True, type=str)
@click.option('-b', '--bit_precision', 'bit_precision', default=23, type=int)
@click.option('-p', '--predictions', 'predictions_output_path', default='/dev/stdout', type=click.Path(exists=False))
@click.option('-t', '--testonly/--learn', 'testonly', default=False)
@click.option('--l2', 'reg', default=2, type=float)
@click.option('-m', '--mode/--sampling', 'mode', default=True)
@click.option('-f', '--final_regressor', 'final_regressor', default='', type=click.Path(exists=False))
@click.option('-i', '--initial_regressor', 'initial_regressor', default='', type=click.Path(exists=False))
@click.option('-P', '--progress', 'progress', callback=validate_progress, default="100")
@click.option('--quiet/--no-quiet', 'quiet', default=False)
@click.option('--input_path', 'input_path', default='/dev/stdin', type=click.Path(exists=False))
def cli(**kwargs):
    bang(**kwargs)


def bang(quadratic_interactions='',
         bit_precision=23,
         testonly=False,
         mode=True,
         reg=2,
         progress=100,
         final_regressor='',
         initial_regressor='',
         quiet=False,
         predictions_output_path='/dev/stdout',
         keep_namespaces=[],
         ignore_namespaces=[],
         input_path='/dev/stdin'):

    output_file = open(predictions_output_path, 'w' if '/dev/stdout' in predictions_output_path else "a+")
    input_file = open(input_path, 'r')

    if not quiet:
        out = get_header(bit_precision, reg, quadratic_interactions, keep_namespaces, ignore_namespaces)
        output_file.write(out)

    # feature_transformer = Spirit(bit_precision)
    model = LogisticBang(bit_precision, reg)
    if initial_regressor:
        model.load(initial_regressor)

    for i, row in enumerate(
            lines_transformer(input_file, quadratic_interactions, bit_precision, set(keep_namespaces),
                              set(ignore_namespaces))):
        if mode:
            prediction = model.predict(row)
        else:
            prediction = model.sample_predict(row)
        out = str(prediction) + "\n"
        if quiet:
            output_file.write(out)

        if not testonly:
            model.partial_fit(row)

        label, weight, _, features = row
        if not i % progress:
            if not quiet:
                out = "%.4f" % model.average_loss + "\t\t" + str(model.example_counter) + "\t\t" + str(weight) + "\t\t"
                out += str(label) + "\t\t" + "%.4f" % prediction + "\t\t" + str(len(features)) + "\n"
                output_file.write(out)
            if isinstance(progress, float):
                progress *= progress

    if final_regressor:
        model.dump(final_regressor)

    output_file.close()
    input_file.close()


if __name__ == "__main__":
    bang()
