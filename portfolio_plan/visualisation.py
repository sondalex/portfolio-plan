from typing import Dict, Literal, TypedDict

from plotnine.geoms import geom_line, geom_text
from plotnine.scales.scale_color import scale_fill_brewer
from plotnine.scales.scale_manual import scale_color_manual, scale_fill_manual
from plotnine.themes.elements import element_line, element_rect, element_text
from plotnine.themes.theme import theme
from plotnine.themes.theme_bw import theme_bw

THEME_ROSE_PINE_BASE_SIZE = 11


class ColorPalette(TypedDict):
    love: str
    gold: str
    rose: str
    pine: str
    foam: str
    iris: str


class Palette(TypedDict):
    background: str
    surface: str
    text: str
    highlight_major: str
    highlight_minor: str
    border: str
    color: ColorPalette


class theme_rose_pine(theme_bw):
    def __init__(
        self,
        variant: Literal["main", "moon", "dawn"] = "main",
        geom_color: Literal["love", "gold", "rose", "pine", "foam", "iris"] = "love",
        base_size: int = THEME_ROSE_PINE_BASE_SIZE,
        base_family=None,
    ):
        """
        Initialize the Rose Pine theme with support for Main, Moon, and Dawn variants.

        Parameters:
        -----------
        variant: str
            "main", "moon", or "dawn" to select the color palette.
        color: str
            Default geom color category.
            See category at `rosé pine https://rosepinetheme.com/palette/`
        base_size: int
            Base font size.
        base_family: str
            Base font family.
        """
        super().__init__(base_size, base_family)

        palettes: Dict[str, Palette] = {
            "main": {
                "background": "#191724",
                "surface": "#1f1d2e",
                "text": "#e0def4",
                "highlight_major": "#eb6f92",
                "highlight_minor": "#f6c177",
                "border": "#6e6a86",
                "color": {
                    "love": "#eb6f92",
                    "gold": "#f6c177",
                    "rose": "#ebbcba",
                    "pine": "#31748f",
                    "foam": "#9ccfd8",
                    "iris": "#c4a7e7",
                },
            },
            "moon": {
                "background": "#232136",
                "surface": "#2a273f",
                "text": "#e0def4",
                "highlight_major": "#eb6f92",
                "highlight_minor": "#f6c177",
                "border": "#6e6a86",
                "color": {
                    "love": "#eb6f92",
                    "gold": "#f6c177",
                    "rose": "#ea9a97",
                    "pine": "#3e8fb0",
                    "foam": "#9ccfd8",
                    "iris": "#c4a7e7",
                },
            },
            "dawn": {
                "background": "#faf4ed",
                "surface": "#fffaf3",
                "text": "#575279",
                "highlight_major": "#b4637a",
                "highlight_minor": "#ea9d34",
                "border": "#9893a5",
                "color": {
                    "love": "#b4637a",
                    "gold": "#ea9d34",
                    "rose": "#d7827e",
                    "pine": "#286983",
                    "foam": "#56949f",
                    "iris": "#907aa9",
                },
            },
        }

        if variant not in palettes:
            raise ValueError(
                f"Invalid variant '{variant}'. Choose from 'main', 'moon', or 'dawn'."
            )

        palette = palettes[variant]
        geom_text.DEFAULT_AES["color"] = palette["text"]
        geom_line.DEFAULT_AES["color"] = palette["color"][geom_color]
        self += theme(  # type: ignore
            text=element_text(color=palette["text"], face="plain"),
            plot_background=element_rect(
                fill=palette["background"], color=palette["border"]
            ),
            plot_title=element_text(color=palette["text"]),
            axis_title=element_text(color=palette["text"]),
            panel_background=element_rect(
                fill=palette["surface"], color=palette["border"]
            ),
            panel_grid_major=element_line(
                color=palette["highlight_major"], size=0.5, linetype="dotted"
            ),
            panel_grid_minor=element_line(
                color=palette["highlight_minor"], size=0.25, linetype="dotted"
            ),
            axis_text=element_text(color=palette["text"]),
            legend_text=element_text(color=palette["text"]),
            legend_title=element_text(color=palette["text"]),
            legend_background=element_rect(fill=palette["background"]),
            legend_key=element_rect(fill=palette["background"], color="none"),
            strip_background=element_rect(
                fill=palette["surface"], color=palette["border"]
            ),
            strip_text_x=element_text(color=palette["text"]),
            strip_text_y=element_text(color=palette["text"], angle=-90),
        )


class scale_rose_pine_discrete(scale_color_manual):
    def __init__(self, variant="main"):
        """
        Custom discrete color scale for the Rose Pine theme.

        Parameters:
        - variant (str): "main", "moon", or "dawn" to select the color palette.
        """
        # Define color palettes for each variant
        palettes = {
            "main": {
                "love": "#eb6f92",
                "gold": "#f6c177",
                "rose": "#ebbcba",
                "pine": "#31748f",
                "foam": "#9ccfd8",
                "iris": "#c4a7e7",
            },
            "moon": {
                "love": "#eb6f92",
                "gold": "#f6c177",
                "rose": "#ea9a97",
                "pine": "#3e8fb0",
                "foam": "#9ccfd8",
                "iris": "#c4a7e7",
            },
            "dawn": {
                "love": "#b4637a",
                "gold": "#ea9d34",
                "rose": "#d7827e",
                "pine": "#286983",
                "foam": "#56949f",
                "iris": "#907aa9",
            },
        }

        if variant not in palettes:
            raise ValueError(
                f"Invalid variant '{variant}'. Choose from 'main', 'moon', or 'dawn'."
            )

        palette = palettes[variant]

        super().__init__(values=list(palette.values()))


class scale_rose_pine_fill_discrete(scale_fill_manual):
    def __init__(self, variant="main"):
        palettes = {
            "main": {
                "love": "#eb6f92",
                "gold": "#f6c177",
                "rose": "#ebbcba",
                "pine": "#31748f",
                "foam": "#9ccfd8",
                "iris": "#c4a7e7",
            },
            "moon": {
                "love": "#eb6f92",
                "gold": "#f6c177",
                "rose": "#ea9a97",
                "pine": "#3e8fb0",
                "foam": "#9ccfd8",
                "iris": "#c4a7e7",
            },
            "dawn": {
                "love": "#b4637a",
                "gold": "#ea9d34",
                "rose": "#d7827e",
                "pine": "#286983",
                "foam": "#56949f",
                "iris": "#907aa9",
            },
        }

        if variant not in palettes:
            raise ValueError(
                f"Invalid variant '{variant}'. Choose from 'main', 'moon', or 'dawn'."
            )

        palette = palettes[variant]
        sorted_values = map(lambda key: palette[key], sorted(palette.keys()))
        super().__init__(values=list(sorted_values))


class scale_brewer_fill_discrete(scale_fill_brewer):
    # TODO: test visibility with variant main, moon, dawn
    def __init__(self, palette="Set3"):
        """
        Parameters
        ----------
        palette: str
            Refer to `ggplot2 https://ggplot2.tidyverse.org/reference/scale_brewer.html`
            and to `ggplot2 book https://ggplot2-book.org/scales-colour#brewer-scales`
        """
        super().__init__(type="qual", palette=palette)


class scale_alphabet_fill_discrete(scale_fill_manual):
    """
    Based on `pals https://kwstat.github.io/pals/` alphabet
    """

    def __init__(self):
        """ """
        values = [
            "#F0A0FF",
            "#0075DC",
            "#993F00",
            "#4C005C",
            "#191919",
            "#005C31",
            "#2BCE48",
            "#FFCC99",
            "#808080",
            "#94FFB5",
            "#8F7C00",
            "#9DCC00",
            "#C20088",
            "#003380",
            "#FFA405",
            "#FFA8BB",
            "#426600",
            "#FF0010",
            "#5EF1F2",
            "#00998F",
            "#E0FF66",
            "#740AFF",
            "#990000",
            "#FFFF80",
            "#FFE100",
            "#FF5005",
        ]
        super().__init__(values=values)
