from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np
import xxhash

from src.utils import *


def hasher(label: str) -> int:
    return xxhash.xxh64_intdigest(label) % hasher.range


def parse_feature(label: str, value: str = 1.0) -> Feature:
    return hasher(label), np.float32(value)


def parse_namespace(namespace: List) -> Tuple[str, Namespace]:
    namespace, *features = namespace.split(" ")
    namespace = namespace[0]
    return namespace, tuple(map(lambda f: parse_feature(*f.split(":")), features))


def parser(row: str, keep_namespaces: Set[str], ignore_namespaces: Set[str]) -> Tuple[
    Float32, Float32, Any, Namespaces]:
    meta, *namespaces = row.split("|")
    meta = meta.strip()
    meta = meta.split(" ")
    if len(meta) == 1:
        label, weight, tag = meta[0], 1, None
    elif len(meta) == 2:
        (label, weight), tag = meta, None
    else:
        label, weight, tag = meta

    # namespaces = dict(parser.pool.map(parse_namespace, namespaces))
    namespaces = dict([parse_namespace(namespace) for namespace in namespaces
                       if not namespace[0]  # not namespace
                       or (
                               (
                                       not keep_namespaces # empty keep_namespaces
                                       or namespace[0] in keep_namespaces # namespace in keep_namespaces
                               )
                               and (
                                       not ignore_namespaces # empty ignore_namespaces
                                       or namespace[0] not in ignore_namespaces # namespace not in ignore_namespace
                               )
                       )])
    return np.float32(label), np.float32(weight), tag, namespaces


def generate_features(namespaces, quadratic_interactions) -> Features:
    bias_feature = generate_bias_feature()
    linear_features = generate_linear_features(namespaces)
    quadratic_features = generate_quadratic_features(quadratic_interactions, namespaces)
    features = bias_feature + linear_features + quadratic_features

    return features


def generate_bias_feature() -> Features:
    return [(-1, np.float32(1))]


def generate_linear_features(namespaces: Namespaces) -> Features:
    return [feature for features in namespaces.values() for feature in features]


def generate_quadratic_features(quadratic_interactions: List[str], namespaces: Namespaces) -> Features:
    features = []
    for interaction in quadratic_interactions:
        namespace1, namespace2 = interaction
        quadratic_namespaces = product(namespaces[namespace1], namespaces[namespace2])
        # features.extend(generate_quadratic_features.pool.map(quadratic_combinator, it))
        features.extend([quadratic_combinator(quadratic_namespace) for quadratic_namespace in quadratic_namespaces])
    return features


def quadratic_combinator(comb: Tuple[Feature, Feature]) -> Feature:
    feature1, feature2 = comb
    label1, value1 = feature1
    label2, value2 = feature2
    return hasher(str(label1) + "^" + str(label2)), value1 * value2


def line_transformer(line: str, quadratic_interactions: List[str], keep_namespaces: Set[str],
                     ignore_namespaces: Set[str]) -> Row:
    parsed_line = parser(line, keep_namespaces, ignore_namespaces)
    label, weight, tag, namespaces = parsed_line

    label = 0 if label == -1 else label
    features = generate_features(namespaces, quadratic_interactions)
    features = np.array(features, dtype=[("id", np.int), ("value", np.float32)])
    return label, weight, tag, features


def lines_transformer(lines: List[str], quadratic_interactions: List[str], bit_precision: int,
                      keep_namespaces: Set[str], ignore_namespaces: Set[str]) -> Rows:
    hasher.range = 2 ** bit_precision

    fn = partial(line_transformer, quadratic_interactions=quadratic_interactions, keep_namespaces=keep_namespaces,
                 ignore_namespaces=ignore_namespaces)
    with Pool(5) as pool:
        for transformed_row in pool.imap(fn, lines):
            label, weight, tag, features = transformed_row
            yield label, weight, tag, features
