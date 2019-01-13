from itertools import product
from multiprocessing import Pool
from typing import List, Tuple, Any, Dict

import click
import numpy as np
import xxhash

Feature = Tuple[int, float]
Namespace = Tuple[Feature]
Namespaces = Dict[str, Namespace]


def hasher(label: str) -> int:
    return xxhash.xxh64_intdigest(label) % hasher.range


def parse_feature(label: str, value: str) -> Feature:
    return hasher(label), float(value)


def parse_namespace(namespace: List) -> Tuple[str, Namespace]:
    namespace, *features = namespace.split(" ")
    return namespace[0], tuple(map(lambda f: parse_feature(*f.split(":")), features))


def parser(row: str) -> Tuple[float, float, Any, Namespaces]:
    meta, *namespaces = row.split("|")
    label, weight, *tag = meta.split(" ")
    namespaces = dict(parser.pool.map(parse_namespace, namespaces))
    return float(label), float(weight), tag, namespaces


def generate_features(namespaces, quadratic_interactions) -> List[Feature]:
    bias_feature = generate_bias_feature()
    linear_features = generate_linear_features(namespaces)
    quadratic_features = generate_quadratic_features(quadratic_interactions, namespaces)
    features = bias_feature + linear_features + quadratic_features

    return features


def generate_bias_feature() -> List[Feature]:
    return [(-1, 1)]


def generate_linear_features(namespaces: Namespaces) -> List[Feature]:
    return [feature for features in namespaces.values() for feature in features]


def generate_quadratic_features(quadratic_interactions: List[str], namespaces: Namespaces) -> List[Feature]:
    features = []
    for comb in quadratic_interactions:
        namespace1, namespace2 = comb
        it = product(namespaces[namespace1], namespaces[namespace2])
        features.extend(generate_quadratic_features.pool.map(quadratic_combinator, it))
    return features


def quadratic_combinator(comb: Tuple[Feature, Feature]) -> Feature:
    feature1, feature2 = comb
    label1, value1 = feature1
    label2, value2 = feature2
    return hasher(str(label1) + "^" + str(label2)), value1 * value2


class LogisticBang:
    def __init__(self, feature_range):
        self.weight = np.zeros((1, feature_range + 1), dtype=np.float32)
        self.Hessian = np.empty((feature_range + 1, feature_range + 1), dtype=np.float32)


click.command()
click.option('-q', '--quadratic', 'quadratic_interactions', default='', type=str)
click.option('-b', '--bit_precision', 'bit_precision', default=15, type=int)


def bang(quadratic_interactions, bit_precision):
    print(quadratic_interactions)
    feature_range = 2 ** bit_precision

    hasher.range = feature_range

    parser.pool = Pool(5)
    generate_quadratic_features.pool = Pool(5)

    row2 = "1 1.0 zebra|MetricFeatures:3.28 height:1.5 length:2.0|etricFeatures2:3.28 height2:1.5 length2:2.0 width2:1|tricFeatures2:3.28 height2:1.5 length2:2.0 width2:1"
    label, weight, tag, namespaces = parser(row2)

    features = generate_features(namespaces, quadratic_interactions)

    ids, values = zip(*features)
    print(ids)
    print(values)


if __name__ == "__main__":
    bang(['Me', 'Mt'], 15)
