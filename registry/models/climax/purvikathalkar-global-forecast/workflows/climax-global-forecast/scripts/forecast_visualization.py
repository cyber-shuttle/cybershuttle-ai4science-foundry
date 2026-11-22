"""Write small dependency-free SVG maps for forecast comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np


_VIRIDIS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)


def _color(value: float, minimum: float, maximum: float) -> str:
    fraction = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
    fraction = min(1.0, max(0.0, fraction))
    position = fraction * (len(_VIRIDIS) - 1)
    left = min(int(position), len(_VIRIDIS) - 2)
    blend = position - left
    rgb = tuple(
        round(_VIRIDIS[left][channel] * (1.0 - blend) + _VIRIDIS[left + 1][channel] * blend)
        for channel in range(3)
    )
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def write_forecast_svg(
    output_path: Path, panels: Sequence[Tuple[str, np.ndarray]]
) -> None:
    if not panels:
        raise ValueError("At least one panel is required")
    arrays = [np.asarray(values, dtype=np.float64) for _, values in panels]
    shape = arrays[0].shape
    if len(shape) != 2 or any(values.shape != shape for values in arrays):
        raise ValueError("All visualization panels must have the same two-dimensional shape")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Visualization panels must contain only finite values")

    minimum = min(float(values.min()) for values in arrays)
    maximum = max(float(values.max()) for values in arrays)
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    panel_width = 350
    panel_height = 215
    grid_width = 320
    grid_height = 160
    cell_width = grid_width / shape[1]
    cell_height = grid_height / shape[0]
    width = columns * panel_width + 40
    height = rows * panel_height + 84

    parts: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        "<title>Forecast comparison maps</title>",
        "<defs><linearGradient id=\"scale\">",
    ]
    for index, rgb in enumerate(_VIRIDIS):
        color = "#{:02x}{:02x}{:02x}".format(*rgb)
        parts.append(
            '<stop offset="{}%" stop-color="{}"/>'.format(
                round(index * 100 / (len(_VIRIDIS) - 1)), color
            )
        )
    parts.extend(
        [
            "</linearGradient></defs>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]
    )

    for index, ((title, _), values) in enumerate(zip(panels, arrays)):
        column = index % columns
        row = index // columns
        origin_x = 20 + column * panel_width
        origin_y = 18 + row * panel_height
        parts.append(
            '<text x="{}" y="{}" font-family="sans-serif" font-size="15" font-weight="600">{}</text>'.format(
                origin_x, origin_y + 16, escape(title)
            )
        )
        grid_x = origin_x
        grid_y = origin_y + 28
        parts.append(
            '<g shape-rendering="crispEdges" aria-label="{}">'.format(escape(title))
        )
        for latitude_index in range(shape[0]):
            for longitude_index in range(shape[1]):
                parts.append(
                    '<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" fill="{}"/>'.format(
                        grid_x + longitude_index * cell_width,
                        grid_y + latitude_index * cell_height,
                        cell_width + 0.02,
                        cell_height + 0.02,
                        _color(float(values[latitude_index, longitude_index]), minimum, maximum),
                    )
                )
        parts.append("</g>")

    legend_y = rows * panel_height + 8
    parts.extend(
        [
            '<text x="20" y="{}" font-family="sans-serif" font-size="12">Shared color scale</text>'.format(
                legend_y + 15
            ),
            '<rect x="145" y="{}" width="250" height="14" fill="url(#scale)"/>'.format(
                legend_y + 3
            ),
            '<text x="405" y="{}" font-family="sans-serif" font-size="12">{:.5g}</text>'.format(
                legend_y + 15, maximum
            ),
            '<text x="145" y="{}" font-family="sans-serif" font-size="12">{:.5g}</text>'.format(
                legend_y + 32, minimum
            ),
            '<text x="20" y="{}" font-family="sans-serif" font-size="11" fill="#555">Grid rows follow the stored latitude order; columns follow longitude order.</text>'.format(
                legend_y + 54
            ),
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
