import numpy as np
from bokeh.plotting import figure, show
from sklearn.metrics import mean_squared_error
from bokeh.models import LinearAxis, Range1d
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
from .geometry import full_hull,lower_hull


def binary_convex_hull_plotter(
    comp,
    observed_energies,
    predicted_energies: np.ndarray = None,
    names: np.ndarray = None,
):
    """Formation energy and convex hull plotter.

    Parameters
    ----------
    comp: np.ndarray
        Matrix of compositons, shape (n,m) of n configurations, m composition dimensions.
    observed_energies: np.ndarray
        Vector of observed formation energies, shape (n,) of n configurations.
    predicted_energies: np.ndarray, optional
        Vector of predicted formation energies, shape (n,) of n configurations.

    Returns
    -------
    p:figurebokeh.plotting._figure.figure
        Bokeh figure object. Can be displayed with bokeh.plotting.show(p)
    """
    # Format data in a dictionary for use in bokeh hover tools
    data_dictionary = {"comp": np.ravel(comp), "observed_energies": observed_energies}
    if predicted_energies is not None:
        data_dictionary["predicted_energies"] = predicted_energies
    if names is not None:
        data_dictionary["names"] = names
    source = ColumnDataSource(data=data_dictionary)

    # Calculate observed hull vertices
    observed_hull = full_hull(compositions=comp, energies=observed_energies)
    observed_vertices, _ = lower_hull(observed_hull)

    # Sort observed vertices
    hullcomps = np.ravel(comp[observed_vertices])
    sorted_obs_verts = observed_vertices[np.argsort(hullcomps)]

    # Extract hull components and energies
    sorted_obs_hullcomps = np.ravel(comp[sorted_obs_verts])
    sorted_obs_hulleng = observed_energies[sorted_obs_verts]

    # Create a figure and format axes and labels
    p = figure(width=1100, height=800)
    p.xaxis.axis_label = "Composition (X)"
    p.yaxis.axis_label = "Formation Energy"
    p.xaxis.axis_label_text_font_size = "21pt"
    p.yaxis.axis_label_text_font_size = "21pt"
    p.xaxis.major_label_text_font_size = "21pt"
    p.yaxis.major_label_text_font_size = "21pt"

    # Scatter plot for observed energies (in black)
    p.scatter(
        sorted_obs_hullcomps,
        sorted_obs_hulleng,
        marker="diamond",
        size=15,
        color="black",
    )
    p.line(
        sorted_obs_hullcomps,
        sorted_obs_hulleng,
        line_width=2,
        color="black",
        legend_label="Hull Vertices (Observed)",
    )
    p.scatter(
        x="comp",
        y="observed_energies",
        size=8,
        color="black",
        marker="x",
        legend_label="Observed Energies",
        source=source,
    )

    if predicted_energies is not None:
        # Calculate predicted hull vertices
        predicted_hull = full_hull(compositions=comp, energies=predicted_energies)
        predicted_vertices, _ = lower_hull(predicted_hull)

        # Sort predicted vertices
        hullcomps = np.ravel(comp[predicted_vertices])
        sorted_pred_verts = predicted_vertices[np.argsort(hullcomps)]

        # Extract predicted hull components and energies
        sorted_pred_hullcomps = np.ravel(comp[sorted_pred_verts])
        sorted_pred_hulleng = predicted_energies[sorted_pred_verts]

        # Plot green squares on missing ground states
        missing_gs = []
        for ov in observed_vertices:
            if ov not in predicted_vertices:
                missing_gs.append(ov)
        p.scatter(
            np.ravel(comp[missing_gs]),
            predicted_energies[missing_gs],
            marker="square",
            color="green",
            size=15,
            alpha=0.5,
            legend_label="Missing",
        )

        # Plot blue squares on spurious ground states
        spurious_gs = []
        for pv in predicted_vertices:
            if pv not in observed_vertices:
                spurious_gs.append(pv)
        p.scatter(
            np.ravel(comp[spurious_gs]),
            predicted_energies[spurious_gs],
            color="blue",
            marker="square",
            size=15,
            alpha=0.5,
            legend_label="Spurious",
        )

        # Plot predicted hull vertices (in red)
        p.scatter(
            sorted_pred_hullcomps,
            sorted_pred_hulleng,
            marker="diamond",
            size=8,
            color="red",
        )
        p.line(
            sorted_pred_hullcomps,
            sorted_pred_hulleng,
            line_width=2,
            color="red",
            legend_label="Hull Vertices (Predicted)",
        )

        # Scatter plot for predicted energies (in red)
        p.scatter(
            x="comp",
            y="predicted_energies",
            size=8,
            color="red",
            marker="y",
            legend_label="Predicted Energies",
            source=source,
        )

    # Add extra data as popup when mouse is over a point
    hover = HoverTool(tooltips=[("Observed Ef", "@observed_energies")])
    if "predicted_energies" in source.data:
        hover.tooltips.append(("Predicted Ef", "@predicted_energies"))
    if "names" in source.data:
        hover.tooltips.append(("name", "@names"))
    p.add_tools(hover)

    return p

