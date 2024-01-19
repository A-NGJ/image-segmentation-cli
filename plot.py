import matplotlib.pyplot as plt
import numpy as np

MILESTONE_COLORS = [
    "#FF7F00",
    "#FDBF6F",
    "#B2DF8A",
    "#33A02C",
    "#6AC3EB",
    "#0099DA",
    "#023852",
    "#F5DC00",
    "#9DADB0",
    "#0C0C0C",
]


def draw_barchart(ax, categories, color_scheme="tab10", **mpl_kwargs):
    """
    Draws a bar chart with primary and subcategories. Each subcategory across primary categories
    has a consistent color.

    :param categories: Dictionary with primary categories as keys and dictionaries of subcategories as values.
    :param color_scheme: A list of colors or a colormap name to use for the bars.
    :param mpl_kwargs: Matplotlib customization options.
    :return: Matplotlib axes.
    """
    # Collect all unique subcategories across all primary categories
    # all_subcategories = set(
    #     subcat for cat in categories.values() for subcat in cat.keys()
    # )
    unique_categories = set()
    all_subcategories = [
        subcat
        for cat in categories.values()
        for subcat in cat.keys()
        if subcat not in unique_categories and (unique_categories.add(subcat) or True)
    ]

    # Assign a color to each subcategory
    if isinstance(color_scheme, str):
        # Use a colormap
        cmap = plt.get_cmap(color_scheme)
        color_map = {subcat: cmap(i) for i, subcat in enumerate(all_subcategories)}
    elif isinstance(color_scheme, list):
        # Use the provided list of colors
        if len(color_scheme) < len(all_subcategories):
            raise ValueError("Not enough colors provided for all subcategories")
        color_map = dict(zip(all_subcategories, color_scheme))
    else:
        raise ValueError(
            "Invalid color_scheme: Must be a colormap name or a list of colors"
        )

    # Number of bars for each primary category
    primary_cat_counts = {cat: len(subcats) for cat, subcats in categories.items()}

    # Initialize the bar positions and colors
    positions = []
    bar_colors = []
    current_position = 0
    for cat, subcats in categories.items():
        for subcat in subcats:
            positions.append(current_position)
            bar_colors.append(color_map[subcat])
            current_position += 1
        current_position += 1  # +1 for spacing between primary categories

    # Extract subcategory values
    values = [value for cat in categories.values() for value in cat.values()]

    # Plot bars with colors
    for pos, value, color in zip(positions, values, bar_colors):
        ax.bar(pos, value, color=color, align="center")

    yticks = ax.get_yticks()
    for y in yticks:
        ax.axhline(y, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    legend_entries = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[subcat])
        for subcat in sorted(all_subcategories)
    ]
    ax.legend(legend_entries, sorted(all_subcategories))

    # Set x-ticks to be in the middle of each primary category group
    xticks_positions = [
        np.mean(positions[i : i + count])
        for i, count in zip(
            [0] + list(np.cumsum(list(primary_cat_counts.values()))[:-1]),
            primary_cat_counts.values(),
        )
    ]
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(list(primary_cat_counts.keys()))

    # Apply any additional matplotlib customization
    ax.set(**mpl_kwargs)

    return ax
