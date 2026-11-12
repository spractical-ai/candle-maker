import numpy as np
from numpy.typing import NDArray

TimeLabels = list[list[tuple[int, str]] | None]

def render(ohlc: NDArray[np.float64], height: int, width: int, use_axis: bool = False, time_labels: TimeLabels | None = None) -> NDArray[np.uint8]: ...
def render_into(ohlc: NDArray[np.float64], out: NDArray[np.uint8], use_axis: bool = False, time_labels: TimeLabels | None = None) -> None: ...
