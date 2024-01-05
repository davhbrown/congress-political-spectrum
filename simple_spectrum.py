import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Scatter
from bokeh.layouts import column
from bokeh.io import curdoc


def _spread_data(
    length: int, random_seed: int = 6, chamber: str = "Senate"
) -> np.ndarray:
    """
    Return random integers for spread purposes in plotting. Called by make_column_data().

    Args
    ----
    length : number of elements to return

    random_seed : default = 6. Set with int for repeatable results.


    Returns
    -------
    1D numpy.ndarray whose length along axis 0 is `length`.

    """
    # Assigns bounds to vertical datapoint spread in plot,
    # based on dataset size. 'House' has more datapoints.
    if chamber == "Senate":
        low = 20
        high = 80
    elif chamber == "House":
        low = 5
        high = 95

    # Randomly spread
    rng = np.random.default_rng(seed=random_seed)
    random_ints = rng.integers(low=low, high=high, size=length) / 100
    return random_ints


def make_column_data(df: pd.DataFrame) -> dict:
    """Prepare dict for ColumnDataSource().

    Uses 'nominate_dim1' as data column and congressperson name from
    the df passed in to create a dict passable to Bokeh's
    `ColumnDataSource()`.

    Args
    ----
    df : pd.DataFrame already filtered by congressional house and party.

    Returns
    -------
    data : dict
    """
    # Do not make column data for empty dfs (independents)
    if df.empty:
        data = {}
        return data
    else:
        chamber = df["chamber"].values[0]
        spread = _spread_data(df.shape[0], chamber=chamber)

        data = dict(
            x=df["nominate_dim1"].values,
            y=spread,
            name=df["bioname"].values,
            state=df["state_abbrev"].values,
        )

        return data


# Load data
cols = ["congress", "chamber", "bioname", "party_code", "nominate_dim1", "state_abbrev"]
# df = pd.read_csv('https://voteview.com/static/data/out/members/HSall_members.csv',
#                  usecols=cols)
df = pd.read_csv("./data/HSall_members.csv", usecols=cols)

# Democrat (D) & Republican (R) party codes
DEMOCRAT = 100
REPUBLICAN = 200
INDEPENDENT = 328

CONGRESSIONAL_SESSION = 117


# Prune df
df = df[df["congress"] == CONGRESSIONAL_SESSION].reset_index(drop=True)
df = df[df["chamber"] != "President"].reset_index(drop=True)

# Split by chamber
house = df[df["chamber"] == "House"]
senate = df[df["chamber"] == "Senate"]

# del df


## Prepare data for plotting
# Split by party
senate_dems = senate[senate["party_code"] == DEMOCRAT]
senate_reps = senate[senate["party_code"] == REPUBLICAN]
senate_independents = senate[senate["party_code"] == INDEPENDENT]

house_dems = house[house["party_code"] == DEMOCRAT]
house_reps = house[house["party_code"] == REPUBLICAN]
house_independents = house[house["party_code"] == INDEPENDENT]

# Dicts for Bokeh
senate_dem_data = make_column_data(senate_dems)
senate_rep_data = make_column_data(senate_reps)
senate_ind_data = make_column_data(senate_independents)

house_dem_data = make_column_data(house_dems)
house_rep_data = make_column_data(house_reps)
house_ind_data = make_column_data(house_independents)

# Make data Bokeh can read & plot
senate_dem_source = ColumnDataSource(data=senate_dem_data)
senate_rep_source = ColumnDataSource(data=senate_rep_data)
senate_ind_source = ColumnDataSource(data=senate_ind_data)

house_dem_source = ColumnDataSource(data=house_dem_data)
house_rep_source = ColumnDataSource(data=house_rep_data)
house_ind_source = ColumnDataSource(data=house_ind_data)


## Plotting stuff

# Hover over tooltip shows name of congressional rep, pulls from ColumnDataSource
TOOLTIPS = [("Name", "@name"), ("State", "@state")]


# SENATE
plot_top = figure(
    title="Senate",
    x_axis_label=None,
    plot_height=300,
    plot_width=1300,
    tooltips=TOOLTIPS,
    toolbar_location=None,
)
plot_top.x_range.start = -1.0
plot_top.x_range.end = 1.0
plot_top.y_range.start = 0.0
plot_top.y_range.end = 1.0
plot_top.xgrid.grid_line_color = None
plot_top.ygrid.grid_line_color = None
plot_top.yaxis.visible = False


# Senate, each party affiliation
glyph = Scatter(
    x="x", y="y", size=7, fill_color="blue", fill_alpha=0.7, line_color=None
)
plot_top.add_glyph(senate_dem_source, glyph)

glyph = Scatter(x="x", y="y", size=7, fill_color="red", fill_alpha=0.7, line_color=None)
plot_top.add_glyph(senate_rep_source, glyph)

glyph = Scatter(
    x="x", y="y", size=7, fill_color="yellow", fill_alpha=0.7, line_color="black"
)
plot_top.add_glyph(senate_ind_source, glyph)

# refline
refline = ColumnDataSource(
    data=dict(x=[0.0, 0.0], y=[0.0, 1.0])
)  # remember it expects them all the same length..
plot_top.line(
    "x", "y", source=refline, line_width=1, line_color="grey", line_dash="dashed"
)


# HOUSE
plot_bottom = figure(
    title="House",
    x_axis_label="Liberal / Conservative Dimension",
    plot_height=300,
    plot_width=1300,
    tooltips=TOOLTIPS,
    toolbar_location=None,
)
plot_bottom.x_range.start = -1.0
plot_bottom.x_range.end = 1.0
plot_bottom.y_range.start = 0.0
plot_bottom.y_range.end = 1.0
plot_bottom.xgrid.grid_line_color = None
plot_bottom.ygrid.grid_line_color = None
plot_bottom.yaxis.visible = False

# House, each party affiliation
glyph = Scatter(
    x="x", y="y", size=7, fill_color="blue", fill_alpha=0.7, line_color=None
)
plot_bottom.add_glyph(house_dem_source, glyph)

glyph = Scatter(x="x", y="y", size=7, fill_color="red", fill_alpha=0.7, line_color=None)
plot_bottom.add_glyph(house_rep_source, glyph)

if len(house_ind_data) >= 1:
    glyph = Scatter(
        x="x", y="y", size=7, fill_color="yellow", fill_alpha=0.7, line_color="black"
    )
    plot_bottom.add_glyph(house_ind_source, glyph)

# refline
plot_bottom.line(
    "x", "y", source=refline, line_width=1, line_color="grey", line_dash="dashed"
)


# Display
layout = column(plot_top, plot_bottom)
curdoc().add_root(layout)
