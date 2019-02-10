import os

import click
from tqdm import tqdm

from src.linbang import LogisticBang
from src.parser import lines_transformer
from src.transformer import Identity


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


def get_parametrization(bit_precision, reg, quadratic_interactions, keep_namespaces, ignore_namespaces):
    out = ''
    if ignore_namespaces:
        out += "Ignoring namespaces beginning with: {}\n".format(' '.join(ignore_namespaces))

    if keep_namespaces:
        out += "Using namespaces beginning with: {}\n".format(' '.join(keep_namespaces - ignore_namespaces))

    out += "Num weight bits = {}\n".format(bit_precision)

    out += "Using initial l2 regularization = {}\n".format(reg)
    if quadratic_interactions:
        out += "Quadratic interactions = {}\n".format(', '.join(quadratic_interactions))
    return out


def get_header():
    out = ''
    out += "{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format("average", "example", "example", "current", "current", "current")
    out += "{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format("loss", "counter", "weight", "label", "predict", "features")
    return out


def get_line(**kwargs):
    out = "{average_loss:.4f}\t\t{example_counter}\t\t{weight:.4f}\t\t{label}\t\t{prediction:.4f}\t\t{len_features}\n"
    out = out.format(**kwargs)
    return out


@click.command()
@click.option('-q', '--quadratic', 'quadratic_interactions', default='', multiple=True, type=str)
@click.option('--keep', 'keep_namespaces', default='', multiple=True, type=str)
@click.option('--ignore', 'ignore_namespaces', default='', multiple=True, type=str)
@click.option('-b', '--bit_precision', 'bit_precision', default=23, type=int)
@click.option('-p', '--predictions', 'predictions_output_path', default='/dev/stdout', type=click.Path(exists=False))
@click.option('-t', '--testonly/--learn', 'testonly', default=False)
@click.option('--profiling/--not-profiling', 'profiling', default=False)
@click.option('--l2', 'reg', default=2, type=float)
@click.option('-m', '--mode/--sampling', 'mode', default=True)
@click.option('-f', '--final_regressor', 'final_regressor', default='', type=click.Path(exists=False))
@click.option('-i', '--initial_regressor', 'initial_regressor', default='', type=click.Path(exists=False))
@click.option('-P', '--progress', 'progress', callback=validate_progress, default="100")
@click.option('--quiet/--not-quiet', 'quiet', default=False)
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
         profiling=False,
         predictions_output_path='/dev/stdout',
         keep_namespaces=[],
         ignore_namespaces=[],
         input_path='/dev/stdin'):
    """

    :param quadratic_interactions:
    :param bit_precision:
    :param testonly:
    :param mode:
    :param reg:
    :param progress:
    :param final_regressor:
    :param initial_regressor:
    :param quiet:
    :param predictions_output_path:
    :param keep_namespaces:
    :param ignore_namespaces:
    :param input_path:
    :return:
    """
    try:
        output_file = open(predictions_output_path, 'w' if '/dev/stdout' in predictions_output_path else "a+")
        input_file = open(input_path, 'r')
        keep_namespaces = set(''.join(keep_namespaces))
        ignore_namespaces = set(''.join(ignore_namespaces))

        if not quiet:
            out = get_parametrization(bit_precision, reg, quadratic_interactions, keep_namespaces, ignore_namespaces)
            output_file.write(out)
            out = get_header()
            output_file.write(out)

        # feature_transformer = Spirit(bit_precision)
        feature_transformer = Identity()
        model = LogisticBang(bit_precision=bit_precision, init_reg=reg, transformer=feature_transformer)
        if initial_regressor:
            model.load(initial_regressor)

        for i, row in tqdm(enumerate(lines_transformer(lines=input_file,
                                                       quadratic_interactions=quadratic_interactions,
                                                       bit_precision=bit_precision,
                                                       keep_namespaces=keep_namespaces,
                                                       ignore_namespaces=ignore_namespaces)
                                     ), disable=not profiling):
            if quiet and not profiling:
                prediction = model.predict(row) if mode else model.sample_predict(row)
                out = "{}\n".format(prediction)
                output_file.write(out)

            elif not i % progress:
                label, weight, _, features = row
                prediction = model.predict(row) if mode else model.sample_predict(row)
                out = get_line(average_loss=model.average_loss, example_counter=model.example_counter,
                               weight=weight, label=label, prediction=prediction, len_features=len(features))
                if profiling:
                    output_file.write("\n")
                output_file.write(out)
                if isinstance(progress, float):
                    progress *= progress

            if not testonly:
                model.partial_fit(row)

        if final_regressor:
            model.save(final_regressor)

    except KeyboardInterrupt:
        if final_regressor:
            model.save(final_regressor)
    finally:
        output_file.close()
        input_file.close()


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    cli()
