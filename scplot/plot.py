from typing import Union, List, Tuple, Callable

import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from holoviews import dim


def __size_legend(size_min, size_max, dot_min, dot_max, size_tick_labels_format, size_ticks):
    # TODO improve
    size_ticks_pixels = np.interp(size_ticks, (size_min, size_max), (dot_min, dot_max))
    size_tick_labels = [size_tick_labels_format.format(x) for x in size_ticks]
    points = hv.Points(
        {'x': np.repeat(0.1, len(size_ticks)), 'y': np.arange(len(size_ticks), 0, -1),
         'size': size_ticks_pixels},
        vdims='size').opts(xaxis=None, color='black', yaxis=None, size=dim('size'))
    labels = hv.Labels(
        {'x': np.repeat(0.2, len(size_ticks)), 'y': np.arange(len(size_ticks), 0, -1),
         'text': size_tick_labels},
        ['x', 'y'], 'text').opts(text_align='left', text_font_size='9pt')
    overlay = (points * labels)
    overlay.opts(width=125, height=int(len(size_ticks) * (dot_max + 12)), xlim=(0, 1),
                 ylim=(0, len(size_ticks) + 1),
                 invert_yaxis=True, shared_axes=False, show_frame=False)
    return overlay


def __bin(df, nbins, coordinate_columns, reduce_function, coordinate_column_to_range=None):
    # replace coordinates with bin
    for view_column_name in coordinate_columns:  # add view column _bin
        values = df[view_column_name].values
        view_column_range = coordinate_column_to_range.get(view_column_name,
                                                           None) if coordinate_column_to_range is not None else None
        column_min = values.min() if view_column_range is None else view_column_range[0]
        column_max = values.max() if view_column_range is None else view_column_range[1]
        df[view_column_name] = np.floor(
            np.interp(values, [column_min, column_max], [0, nbins - 1])).astype(int)

    agg_func = {}
    for column in df:
        if column == 'count':
            agg_func[column] = 'sum'
        elif str(df[column].dtype) == 'category':
            agg_func[column] = lambda x: x.mode()[0]
        elif column not in coordinate_columns:
            agg_func[column] = reduce_function
    return df.groupby(coordinate_columns, as_index=False).agg(agg_func), df[coordinate_columns]


def violin(adata: AnnData, keys: Union[str, List[str], Tuple[str]], by: str = None,
           width: int = 200, cmap: Union[str, List[str], Tuple[str]] = 'Category20', cols: int = 3,
           use_raw: bool = None, **kwds):
    """
    Generate a violin plot.

    Parameters:
    adata: Annotated data matrix.
    keys: Keys for accessing variables of adata.var_names or fields of adata.obs
    by: Group plot by specified observation.
    width: Plot width.
    cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
    cols: Number of columns for laying out multiple plots
    use_raw: Use `raw` attribute of `adata` if present.
    """

    adata_raw = adata
    if use_raw or (use_raw is None and adata.raw is not None):
        if adata.raw is None:
            raise ValueError('Raw data not found')
        adata_raw = adata.raw
    plots = []
    keywords = dict(padding=0.02, cmap=cmap)
    keywords.update(kwds)
    if not isinstance(keys, (list, tuple, np.ndarray)):
        keys = [keys]
    for key in keys:
        if key in adata_raw.var_names:
            X = adata_raw[:, key].X
            if scipy.sparse.issparse(X):
                X = X.toarray()
            df = pd.DataFrame(X, columns=[key])
            if by is not None:
                df[by] = adata.obs[by].values
        else:
            df = adata.obs
        if by is not None and str(df[by].dtype) == 'category':
            df[by] = df[by].astype(str)  # hvplot does not currently handle categorical type for colors
        p = df.hvplot.violin(key, width=width, by=by, violin_color=by, **keywords)
        plots.append(p)

    return hv.Layout(plots).cols(cols)


def heatmap(adata: AnnData, keys: Union[str, List[str], Tuple[str]], by: str,
            reduce_function: Callable[[np.ndarray], float] = np.mean,
            use_raw: bool = None, cmap: Union[str, List[str], Tuple[str]] = 'Reds', **kwds):
    """
    Generate a heatmap.

    Parameters:
    adata: Annotated data matrix.
    keys: Keys for accessing variables of adata.var_names
    by: Group plot by specified observation.
    cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
    reduce_function: Function to summarize an element in the heatmap
    use_raw: Use `raw` attribute of `adata` if present.
    """

    adata_raw = adata
    if use_raw or (use_raw is None and adata.raw is not None):
        if adata.raw is None:
            raise ValueError('Raw data not found')
        adata_raw = adata.raw
    if not isinstance(keys, (list, tuple, np.ndarray)):
        keys = [keys]
    df = None
    keywords = dict(colorbar=True, xlabel='', cmap=cmap, ylabel=str(by), rot=90)

    keywords.update(kwds)
    for key in keys:
        X = adata_raw[:, key].X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        _df = pd.DataFrame(X, columns=['value'])
        _df['feature'] = key
        _df[by] = adata.obs[by].values
        df = _df if df is None else pd.concat((df, _df))

    return df.hvplot.heatmap(x='feature', y=by, C='value', reduce_function=reduce_function, **keywords)


def scatter(adata: AnnData, x: str, y: str, color=None, size: Union[int, str] = None,
            dot_min=2, dot_max=14, use_raw: bool = None, sort: bool = True, width: int = 400, height: int = 400,
            nbins: int = None, reduce_function: Callable[[np.array], float] = np.mean,
            cmap: Union[str, List[str], Tuple[str]] = 'viridis', **kwds):
    """
    Generate a scatter plot.

    Parameters:
    adata: Annotated data matrix.
    x: Key for accessing variables of adata.var_names or field of adata.obs
    y: Key for accessing variables of adata.var_names or field of adata.obs
    cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
    color: Field in .var_names or adata.obs to color the points by.
    sort: Plot higher color by values on top of lower values.
    size: Field in .var_names or adata.obs to size the points by or a pixel size.
    dot_min: Minimum dot size when sizing points by a field.
    dot_max: Maximum dot size when sizing points by a field.
    use_raw: Use `raw` attribute of `adata` if present.
    nbins: Number of bins used to summarize plot on a grid. Useful for large datasets.
    reduce_function: Function used to summarize overlapping cells if nbins is specified
    """

    adata_raw = adata
    if use_raw or (use_raw is None and adata.raw is not None):
        if adata.raw is None:
            raise ValueError('Raw data not found')
        adata_raw = adata.raw
    # color can be obs (by) or var_name (c)
    keywords = dict(fontsize=dict(title=9), nonselection_alpha=0.1, padding=0.02, xaxis=True, yaxis=True, width=width,
                    height=height, alpha=1, tools=['box_select'], cmap=cmap)
    keywords.update(kwds)
    df = pd.DataFrame(index=adata.obs.index)
    keys = [x, y]
    if color is not None:
        keys.append(color)
    is_size_by = isinstance(size, str)
    if is_size_by:
        keys.append(size)

    # create df
    for key in keys:
        if key in adata_raw.var_names:
            X = adata_raw[:, key].X
            if scipy.sparse.issparse(X):
                X = X.toarray()
            df[key] = X
        else:
            df[key] = adata.obs[key].values

    df_with_coords = df
    if nbins is not None:
        df['count'] = 1.0
        hover_cols = keywords.get('hover_cols', [])
        hover_cols.append('count')
        keywords['hover_cols'] = hover_cols
        df, df_with_coords = __bin(df, nbins=nbins, coordinate_columns=[x, y], reduce_function=reduce_function)

    if color is not None:
        is_color_by_numeric = pd.api.types.is_numeric_dtype(df[color])
        if is_color_by_numeric:
            keywords.update(dict(colorbar=True, c=color))
            if sort:
                df = df.sort_values(by=color)
        else:
            keywords.update(dict(by=color))

    if is_size_by:
        size_min = df[size].min()
        size_max = df[size].max()
        size_pixels = np.interp(df[size], (size_min, size_max), (dot_min, dot_max))
        df['pixels'] = size_pixels
        keywords['s'] = 'pixels'
        hover_cols = keywords.get('hover_cols', [])
        hover_cols.append(size)
        keywords['hover_cols'] = hover_cols
    elif size is not None:
        keywords['size'] = size

    p = df.hvplot.scatter(x=x, y=y, **keywords)
    if is_size_by:
        layout = (p + __size_legend(size_min=size_min, size_max=size_max, dot_min=dot_min, dot_max=dot_max,
                                    size_tick_labels_format='{0:.1f}',
                                    size_ticks=np.array([size_min, (size_min + size_max) / 2, size_max])))
    else:
        layout = hv.Layout([p]).cols(1)
    layout.df = df_with_coords
    return layout


def dotplot(adata: AnnData, keys: Union[str, List[str], Tuple[str]], by: str, reduce_function=np.mean,
            fraction_min: float = 0, fraction_max: float = None, dot_min: int = 0, dot_max: int = 14,
            use_raw: bool = None, cmap: Union[str, List[str], Tuple[str]] = 'Reds', **kwds):
    """
    Generate a dot plot.

    Parameters:
    adata: Annotated data matrix.
    keys: Keys for accessing variables of adata.var_names
    by: Group plot by specified observation.
    cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
    reduce_function: Function to summarize an element in the heatmap
    fraction_min: Minimum fraction expressed value.
    fraction_max: Maximum fraction expressed value.
    dot_min: Minimum pixel dot size.
    dot_max: Maximum pixel dot size.
    use_raw: Use `raw` attribute of `adata` if present.
    """

    adata_raw = adata
    if use_raw or (use_raw is None and adata.raw is not None):
        if adata.raw is None:
            raise ValueError('Raw data not found')
        adata_raw = adata.raw
    if not isinstance(keys, (list, tuple, np.ndarray)):
        keys = [keys]
    keywords = dict(colorbar=True, ylabel=str(by), xlabel='', hover_cols=['fraction'], padding=0, rot=90, cmap=cmap)

    keywords.update(kwds)
    X = adata_raw[:, keys].X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    df = pd.DataFrame(data=X, columns=keys)
    df[by] = adata.obs[by].values

    def non_zero(g):
        return np.count_nonzero(g) / g.shape[0]

    summarized = df.groupby(by).aggregate([reduce_function, non_zero])
    mean_columns = []
    frac_columns = []
    for i in range(len(summarized.columns)):
        if i % 2 == 0:
            mean_columns.append(summarized.columns[i])
        else:
            frac_columns.append(summarized.columns[i])
    fraction_df = summarized[frac_columns]
    mean_df = summarized[mean_columns]

    y, x = np.indices(mean_df.shape)
    y = y.flatten()
    x = x.flatten()
    fraction = fraction_df.values.flatten()
    if fraction_max is None:
        fraction_max = fraction.max()
    size = np.interp(fraction, (fraction_min, fraction_max), (dot_min, dot_max))
    summary_values = mean_df.values.flatten()
    # xlabel = [keys[i] for i in range(len(keys))]
    # ylabel = [str(summarized.index[i]) for i in range(len(summarized.index))]
    tmp_df = pd.DataFrame(
        data=dict(x=x, y=y, value=summary_values, pixels=size, fraction=fraction))

    xticks = [(i, keys[i]) for i in range(len(keys))]
    yticks = [(i, str(summarized.index[i])) for i in range(len(summarized.index))]

    keywords['width'] = int(np.ceil(dot_max * len(xticks) + 150))
    keywords['height'] = int(np.ceil(dot_max * len(yticks) + 100))
    try:
        import bokeh.models
        keywords['tools'] = [bokeh.models.HoverTool(tooltips=[
            ('fraction', '@fraction'),
            ('value', '@value')
        ])]
    except ModuleNotFoundError:
        pass

    p = tmp_df.hvplot.scatter(x='x', y='y', xlim=(-0.5, len(xticks) + 0.5), ylim=(-0.5, len(yticks) + 0.5),
                              c='value', s='pixels', xticks=xticks, yticks=yticks, **keywords)

    size_range = fraction_max - fraction_min
    if 0.3 < size_range <= 0.6:
        size_legend_step = 0.1
    elif size_range <= 0.3:
        size_legend_step = 0.05
    else:
        size_legend_step = 0.2

    size_ticks = np.arange(fraction_min if fraction_min > 0 or fraction_min > 0 else fraction_min + size_legend_step,
                           fraction_max + size_legend_step, size_legend_step)
    return p + __size_legend(size_min=fraction_min, size_max=fraction_max, dot_min=dot_min, dot_max=dot_max,
                             size_tick_labels_format='{:.0%}', size_ticks=size_ticks)


def scatter_matrix(adata: AnnData, keys: Union[str, List[str], Tuple[str]], color=None, use_raw: bool = None, **kwds):
    """
    Generate a scatter plot matrix.

    Parameters:
    adata: Annotated data matrix.
    keys: Key for accessing variables of adata.var_names or a field of adata.obs
    color: Key in adata.obs to color points by.
    use_raw: Use `raw` attribute of `adata` if present.
    """

    adata_raw = adata
    if use_raw or (use_raw is None and adata.raw is not None):
        if adata.raw is None:
            raise ValueError('Raw data not found')
        adata_raw = adata.raw
    if not isinstance(keys, (list, tuple, np.ndarray)):
        keys = [keys]

    df = pd.DataFrame(index=adata.obs.index)
    if color is not None:
        if color in adata_raw.var_names:
            raise ValueError('Coloring scatter matrix by gene expression not yet supported')
            # X = adata_raw[:, c].X
            # if scipy.sparse.issparse(X):
            #     X = X.toarray()
            # df[c] = X
        else:
            df[color] = adata.obs[color].values
    for key in keys:
        if key in adata_raw.var_names:
            X = adata_raw[:, key].X
            if scipy.sparse.issparse(X):
                X = X.toarray()
            df[key] = X
        else:
            df[key] = adata.obs[key].values

    return hvplot.scatter_matrix(df, c=color, **kwds)


def embedding(adata: AnnData, basis: str, keys: Union[None, str, List[str], Tuple[str]] = None,
              cmap: Union[str, List[str], Tuple[str]] = 'viridis',
              alpha: float = 1, size: int = 12,
              width: int = 400, height: int = 400,
              sort: bool = True, cols: int = 2,
              use_raw: bool = None, nbins: int = None, reduce_function: Callable[[np.array], float] = np.mean,
              **kwds):
    """
    Generate an embedding plot.

    Parameters:
    adata: Annotated data matrix.
    keys: Key for accessing variables of adata.var_names or a field of adata.obs used to color the plot. Can also use `count` to plot cell count when binning.
    basis: String in adata.obsm containing coordinates.
    alpha: Points alpha value.
    size: Point pixel size.
    sort: Plot higher values on top of lower values.
    cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
    nbins: Number of bins used to summarize plot on a grid. Useful for large datasets.
    reduce_function: Function used to summarize overlapping cells if nbins is specified.
    cols: Number of columns for laying out multiple plots
    width: Plot width.
    height: Plot height.
    use_raw: Use `raw` attribute of `adata` if present.
    """

    if keys is None:
        keys = []

    adata_raw = adata
    if use_raw or (use_raw is None and adata.raw is not None):
        if adata.raw is None:
            raise ValueError('Raw data not found')
        adata_raw = adata.raw
    if not isinstance(keys, (list, tuple, np.ndarray)):
        keys = [keys]
    keywords = dict(fontsize=dict(title=9), padding=0.02, xaxis=False, yaxis=False, nonselection_alpha=0.1,
                    tools=['box_select'], cmap=cmap)
    keywords.update(kwds)
    coordinate_columns = ['X_' + basis + c for c in ['1', '2']]
    df = pd.DataFrame(adata.obsm['X_' + basis][:, 0:2], columns=coordinate_columns)
    # create df
    for key in keys:
        if key is not 'count':  # count means plot cell count
            if key in adata_raw.var_names:
                X = adata_raw[:, key].X
                if scipy.sparse.issparse(X):
                    X = X.toarray()
                df[key] = X
            else:
                df[key] = adata.obs[key].values

    if len(keys) == 0:
        keys = ['count']
    plots = []
    df_with_coords = df
    if nbins is not None:
        df['count'] = 1.0
        hover_cols = keywords.get('hover_cols', [])
        hover_cols.append('count')
        keywords['hover_cols'] = hover_cols
        df, df_with_coords = __bin(df, nbins=nbins, coordinate_columns=coordinate_columns,
                                   reduce_function=reduce_function)
    for key in keys:
        is_color_by_numeric = pd.api.types.is_numeric_dtype(df[key])
        df_to_plot = df
        if sort and is_color_by_numeric:
            df_to_plot = df.sort_values(by=key)

        p = df_to_plot.hvplot.scatter(x=coordinate_columns[0],
                                      y=coordinate_columns[1],
                                      title=str(key),
                                      c=key if is_color_by_numeric else None,
                                      by=key if not is_color_by_numeric else None,
                                      size=size,
                                      alpha=alpha,
                                      colorbar=is_color_by_numeric,
                                      width=width, height=height, **keywords)
        plots.append(p)
    layout = hv.Layout(plots).cols(cols)
    layout.df = df_with_coords
    return layout
