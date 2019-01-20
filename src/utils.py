from typing import List, Tuple, Any, Dict

from numpy import ndarray, float32

Feature = Tuple[int, float32]
Features = List[Feature]
Namespace = Tuple[Feature]
Namespaces = Dict[str, Namespace]
Array = ndarray
Row = Tuple[float32, float32, Any, Array]
Rows = List[Row]
Float32 = float32
