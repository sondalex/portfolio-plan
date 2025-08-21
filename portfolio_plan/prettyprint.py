"""
Copy pasted from
https://gist.github.com/sondalex/d4d76c44983b485c53cb777f71ca4185
"""

from dataclasses import dataclass
from typing import Any, Dict, List


class Box:
    def __init__(self, data: Any):
        super().__setattr__(
            "_mutable",
            (
                "top_connected",
                "bottom_connected",
                "data",
            ),
        )
        self.data = data
        self.top_connected = False
        self.bottom_connected = False

    def __str__(self):
        ltop = "┌"
        rtop = "┐"
        lbottom = "└"
        rbottom = "┘"
        vbar = "│"
        hbar = "─"
        lines = str(self.data).split("\n")
        max_length = max([len(line) for line in lines], default=0)
        nspace = 2
        total_width = nspace + max_length

        if self.top_connected:
            middle = total_width // 2
            topline = (
                ltop + hbar * middle + "┴" + hbar * (total_width - middle - 1) + rtop
            )
        else:
            topline = ltop + hbar * total_width + rtop

        if self.bottom_connected:
            middle = total_width // 2
            bottomline = (
                lbottom
                + hbar * middle
                + "┬"
                + hbar * (total_width - middle - 1)
                + rbottom
            )
        else:
            bottomline = lbottom + hbar * total_width + rbottom

        content = ""
        for i, (line) in enumerate(lines):
            extraspace = " " * (max_length - len(line))
            content += f"{vbar} {line} {extraspace}{vbar}"
            if i < len(lines) - 1:
                content += "\n"

        return f"{topline}\n{content}\n{bottomline}"

    def __repr__(self):
        return self.__str__()

    def __setattr__(self, name, value):
        if name == "_mutable":
            super().__setattr__(name, value)
            return

        if hasattr(self, "_mutable") and name not in self._mutable:
            raise ValueError(f"Cannot change {name}. Only {self._mutable} are settable")

        if name in self._mutable and name != "data" and not isinstance(value, bool):
            raise ValueError("Value must be a boolean")

        super().__setattr__(name, value)


@dataclass
class Position:
    x: int
    y: int
    width: int
    height: int = 3


def calculate_positions(root_box: Box, child_boxes: List[Box]) -> Dict[Box, Position]:
    positions = {}

    def get_box_width(box: Box) -> int:
        lines = str(box.data).split("\n")  # Get raw data lines
        max_line_length = max([len(line) for line in lines], default=0)
        return max_line_length + 4

    root_width = get_box_width(root_box)
    total_children_width = (
        sum(get_box_width(box) for box in child_boxes) + (len(child_boxes) - 1) * 4
    )
    start_x = max(0, (total_children_width - root_width) // 2)

    positions[root_box] = Position(x=start_x, y=0, width=root_width)

    current_x = 0
    for i, box in enumerate(child_boxes):
        box_width = get_box_width(box)
        positions[box] = Position(x=current_x, y=4, width=box_width)
        current_x += box_width + 4

        box.top_connected = True
        if i == len(child_boxes) // 2:
            root_box.bottom_connected = True

    return positions


def connect(root_box: Box, child_boxes: List[Box]) -> str:
    positions = calculate_positions(root_box, child_boxes)

    max_width = max(pos.x + pos.width for pos in positions.values())
    max_height = max(pos.y + pos.height for pos in positions.values()) + 1 + 1

    grid = [[" " for _ in range(max_width + 1)] for _ in range(max_height)]

    for box, pos in positions.items():
        box_lines = str(box).split("\n")

        y_offset = 1 if box in child_boxes else 0

        for i, line in enumerate(box_lines):
            for j, char in enumerate(line):
                if pos.y + i + y_offset < len(grid) and pos.x + j < len(grid[0]):
                    grid[pos.y + i + y_offset][pos.x + j] = char

    if root_box.bottom_connected and child_boxes:
        root_pos = positions[root_box]
        connection_x = root_pos.x + root_pos.width // 2

        grid[root_pos.y + root_pos.height][connection_x] = "│"

        grid[root_pos.y + root_pos.height + 1][connection_x] = "┴"

        first_node_pos = positions[child_boxes[0]]
        last_node_pos = positions[child_boxes[-1]]
        left_x = first_node_pos.x + (first_node_pos.width // 2)
        right_x = last_node_pos.x + (last_node_pos.width // 2)

        grid[root_pos.y + root_pos.height + 1][left_x] = "┌"
        grid[root_pos.y + root_pos.height + 1][right_x] = "┐"

        for x in range(left_x + 1, right_x):
            if x != connection_x:
                grid[root_pos.y + root_pos.height + 1][x] = "─"

        for child_box in child_boxes[1:-1]:
            child_pos = positions[child_box]
            child_x = child_pos.x + (child_pos.width // 2)
            if child_x != connection_x:
                grid[root_pos.y + root_pos.height + 1][child_x] = "┬"
                pass

    return "\n".join("".join(row).rstrip() for row in grid)


def ascii_graph(data: Dict[str, str], nodes: List[Dict[str, str]]) -> str:
    data_box = Box(data)
    nodes_boxes = [Box(node) for node in nodes]
    return connect(data_box, nodes_boxes)
