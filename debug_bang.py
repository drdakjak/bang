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

    rows = [
        "0 |f 20:4.9423851e-02 54:5.3187788e-02 55:6.0427137e-02 57:1.0564880e-01 73:5.2399613e-02 102:7.7108525e-02 166:1.1860829e-01 170:9.4944857e-02 229:1.3837518e-01 236:6.6884279e-02 240:7.0336357e-02 269:6.9057822e-02 365:7.3943853e-02 388:8.8355467e-02 464:1.0471541e-01 524:1.1728050e-01 535:1.6337742e-01 555:7.6253511e-02 679:1.0454474e-01 724:7.9615399e-02 769:2.7135062e-01 802:1.1741807e-01 1060:1.9710100e-01 1094:2.2201163e-01 1099:1.1616483e-01 1102:1.1104076e-01 1127:6.3278586e-02 1340:1.3284346e-01 1455:1.1657125e-01 1629:8.4958471e-02 2234:2.3131107e-01 2298:4.2629573e-02 2533:1.5628998e-01 3434:1.5262622e-01 4518:2.4061942e-01 4965:1.6932878e-01 7330:3.1824017e-01 11017:1.8524769e-01 15621:5.0792605e-01"]
    rows *= 200

    rows_iterator = lines_transformer(rows, quadratic_interactions, bit_precision)
    row = next(rows_iterator)
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
