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

    # rows = [
    #     "1 0.0001 zebra|MetricFeatures:3.28 height:1.5 length:2.0|etricFeatures2:3.28 height2:1.5 length2:2.0 width2:1|tricFeatures2:3.28 height2:1.5 length2:2.0 width2:1",
    #     "0 1.0 zebra|MetricFeatures:3.28 height:1.5 length:2.0|etricFeatures2:3.28 height2:1.5 length2:2.0 width2:1|tricFeatures2:3.28 height2:1.5 length2:2.0 width2:1"]
    # rows *= 200
    # rows_iterator = lines_transformer(, quadratic_interactions)

    # model.fit(rows_iterator)

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
