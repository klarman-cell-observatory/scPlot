from typing import Union, List, Tuple, Callable

import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from holoviews import dim


def __create_hover_tool(df, keywords, exclude, current):
    try:
        import bokeh.models
        import holoviews.core.util
        hover_cols = []
        for column in df.columns:
            if column not in exclude and column != current and column not in hover_cols:
                hover_cols.append(column)
        keywords['hover_cols'] = hover_cols
        tooltips = []
        tooltips.append((current, '@{' + holoviews.core.util.dimension_sanitizer(current) + '}'))
        for hover_col in hover_cols:
            tooltips.append((hover_col, '@{' + holoviews.core.util.dimension_sanitizer(hover_col) + '}'))
        keywords['tools'] = [bokeh.models.HoverTool(tooltips=tooltips)]
    except ModuleNotFoundError:
        pass


def __create_bounds_stream(source):
    stream = hv.streams.BoundsXY(source=source)
    return stream


def get_bounds(plot):
    if isinstance(plot, hv.Layout):
        if plot.shape == (1, 1):
            plot = plot[0, 0]
        else:
            raise ValueError('Please select the plot in the layout')
    if hasattr(plot, 'bounds_stream'):
        return plot.bounds_stream.bounds


def __to_list(vals):
    if isinstance(vals, np.ndarray):
        vals = vals.tolist()
    elif isinstance(vals, tuple):
        vals = list(vals)
    elif not isinstance(vals, list):
        vals = [vals]
    return vals


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


def __fix_color_by_data_type(df, by):
    if by is not None and (pd.api.types.is_categorical_dtype(df[by]) or pd.api.types.is_bool_dtype(df[by])):
        df[by] = df[by].astype(str)  # hvplot does not currently handle categorical or boolean type for colors


def __get_raw(adata, use_raw):
    adata_raw = adata
    if use_raw or (use_raw is None and adata.raw is not None):
        if adata.raw is None:
            raise ValueError('Raw data not found')
        adata_raw = adata.raw
    return adata_raw


def __get_df(adata, adata_raw, keys, df=None, is_obs=None):
    if df is not None and is_obs is None:
        raise ValueError('Please provide is_obs when df is provided.')
    for key in keys:
        if df is None:
            is_obs = key not in adata.var
            df = pd.DataFrame(data=dict(id=(adata.obs.index.values if is_obs else adata.var.index.values)))
        if key in adata_raw.var_names and is_obs:
            X = adata_raw[:, key].X
            if scipy.sparse.issparse(X):
                X = X.toarray()
            df[key] = X
        elif key in adata.obs and is_obs:
            df[key] = adata.obs[key].values
        elif key in adata.var and not is_obs:
            df[key] = adata.var[key].values
        else:
            raise ValueError('{} not found'.format(key))
    return df


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
        elif pd.api.types.is_categorical_dtype(df[column]):
            agg_func[column] = lambda x: x.mode()[0]
        elif column not in coordinate_columns and pd.api.types.is_numeric_dtype(df[column]):
            agg_func[column] = reduce_function
    return df.groupby(coordinate_columns, as_index=False).agg(agg_func), df[coordinate_columns]


def violin(adata: AnnData, keys: Union[str, List[str], Tuple[str]], by: str = None,
           width: int = 200, cmap: Union[str, List[str], Tuple[str]] = 'Category20', cols: int = 3,
           use_raw: bool = None, **kwds):
    """
    Generate a violin plot.

    Parameters:
    adata: Annotated data matrix.
    keys: Keys for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
    by: Group plot by specified observation.
    width: Plot width.
    cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
    cols: Number of columns for laying out multiple plots
    use_raw: Use `raw` attribute of `adata` if present.
    """

    adata_raw = __get_raw(adata, use_raw)

    plots = []
    keywords = dict(padding=0.02, cmap=cmap, rot=90)
    keywords.update(kwds)
    keys = __to_list(keys)
    df = __get_df(adata, adata_raw, keys + ([] if by is None else [by]))
    __fix_color_by_data_type(df, by)
    for key in keys:
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

    adata_raw = __get_raw(adata, use_raw)
    keys = __to_list(keys)
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
    x: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
    y: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
    cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
    color: Field in .var_names, adata.var, or adata.obs to color the points by.
    sort: Plot higher color by values on top of lower values.
    size: Field in .var_names, adata.var, or adata.obs to size the points by or a pixel size.
    dot_min: Minimum dot size when sizing points by a field.
    dot_max: Maximum dot size when sizing points by a field.
    use_raw: Use `raw` attribute of `adata` if present.
    nbins: Number of bins used to summarize plot on a grid. Useful for large datasets.
    reduce_function: Function used to summarize overlapping cells if nbins is specified
    """
    return __scatter(adata=adata, x=x, y=y, color=color, size=size, dot_min=dot_min, dot_max=dot_max, use_raw=use_raw,
        sort=sort, width=width, height=height, nbins=nbins, reduce_function=reduce_function, cmap=cmap, is_scatter=True,
        **kwds)


def line(adata: AnnData, x: str, y: str,
         use_raw: bool = None, width: int = 400, height: int = 400,
         nbins: int = None, reduce_function: Callable[[np.array], float] = np.mean, **kwds):
    """
    Generate a scatter plot.

    Parameters:
    adata: Annotated data matrix.
    x: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
    y: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
    use_raw: Use `raw` attribute of `adata` if present.
    nbins: Number of bins used to summarize plot on a grid. Useful for large datasets.
    reduce_function: Function used to summarize overlapping cells if nbins is specified
    """
    return __scatter(adata=adata, x=x, y=y, use_raw=use_raw, sort=False, width=width, height=height, nbins=nbins,
        reduce_function=reduce_function, is_scatter=False,
        **kwds)


def __scatter(adata: AnnData, x: str, y: str, color=None, size: Union[int, str] = None,
              dot_min=2, dot_max=14, use_raw: bool = None, sort: bool = True, width: int = 400, height: int = 400,
              nbins: int = None, reduce_function: Callable[[np.array], float] = np.mean,
              cmap: Union[str, List[str], Tuple[str]] = 'viridis', is_scatter=True, **kwds):
    """
    Generate a scatter plot.

    Parameters:
    adata: Annotated data matrix.
    x: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
    y: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
    cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
    color: Field in .var_names, adata.var, or adata.obs to color the points by.
    sort: Plot higher color by values on top of lower values.
    size: Field in .var_names, adata.var, or adata.obs to size the points by or a pixel size.
    dot_min: Minimum dot size when sizing points by a field.
    dot_max: Maximum dot size when sizing points by a field.
    use_raw: Use `raw` attribute of `adata` if present.
    nbins: Number of bins used to summarize plot on a grid. Useful for large datasets.
    reduce_function: Function used to summarize overlapping cells if nbins is specified
    """

    adata_raw = __get_raw(adata, use_raw)
    keywords = dict(fontsize=dict(title=9), nonselection_alpha=0.1, padding=0.02, xaxis=True, yaxis=True, width=width,
        height=height, alpha=1, tools=['box_select'], cmap=cmap)
    keywords.update(kwds)

    keys = [x, y]
    if color is not None and is_scatter:
        keys.append(color)
    is_size_by = isinstance(size, str)
    if is_size_by and is_scatter:
        keys.append(size)

    df = __get_df(adata, adata_raw, keys)
    df_with_coords = df
    hover_cols = keywords.get('hover_cols', [])
    if nbins is not None:
        df['count'] = 1.0
        hover_cols.append('count')
        df, df_with_coords = __bin(df, nbins=nbins, coordinate_columns=[x, y], reduce_function=reduce_function)
    else:
        hover_cols.append('id')
    keywords['hover_cols'] = hover_cols
    if color is not None and is_scatter:
        __fix_color_by_data_type(df, color)
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
    if is_scatter:
        p = df.hvplot.scatter(x=x, y=y, **keywords)
    else:
        df = df.sort_values(by=x)
        p = df.hvplot.line(x=x, y=y, **keywords)

    if is_size_by:
        return_value = (p + __size_legend(size_min=size_min, size_max=size_max, dot_min=dot_min, dot_max=dot_max,
            size_tick_labels_format='{0:.1f}',
            size_ticks=np.array([size_min, (size_min + size_max) / 2, size_max])))
    else:
        return_value = p
    return_value.df = df_with_coords
    return return_value


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

    adata_raw = __get_raw(adata, use_raw)
    keys = __to_list(keys)
    keywords = dict(colorbar=True, ylabel=str(by), xlabel='', padding=0, rot=90, cmap=cmap)

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
    xlabel = [keys[i] for i in range(len(keys))]
    ylabel = [str(summarized.index[i]) for i in range(len(summarized.index))]
    dotplot_df = pd.DataFrame(
        data=dict(x=x, y=y, value=summary_values, pixels=size, fraction=fraction, xlabel=np.array(xlabel)[x],
            ylabel=np.array(ylabel)[y]))

    xticks = [(i, keys[i]) for i in range(len(keys))]
    yticks = [(i, str(summarized.index[i])) for i in range(len(summarized.index))]

    keywords['width'] = int(np.ceil(dot_max * len(xticks) + 150))
    keywords['height'] = int(np.ceil(dot_max * len(yticks) + 100))
    try:
        import bokeh.models
        keywords['hover_cols'] = ['fraction', 'xlabel', 'ylabel']
        keywords['tools'] = [bokeh.models.HoverTool(tooltips=[
                ('fraction', '@fraction'),
                ('value', '@value'),
                ('x', '@xlabel'),
                ('y', '@ylabel')
        ])]
    except ModuleNotFoundError:
        pass

    p = dotplot_df.hvplot.scatter(x='x', y='y', xlim=(-0.5, len(xticks) + 0.5), ylim=(-0.5, len(yticks) + 0.5),
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
    result = p + __size_legend(size_min=fraction_min, size_max=fraction_max, dot_min=dot_min, dot_max=dot_max,
        size_tick_labels_format='{:.0%}', size_ticks=size_ticks)
    result.df = dotplot_df
    return result


def scatter_matrix(adata: AnnData, keys: Union[str, List[str], Tuple[str]], color=None, use_raw: bool = None, **kwds):
    """
    Generate a scatter plot matrix.

    Parameters:
    adata: Annotated data matrix.
    keys: Key for accessing variables of adata.var_names or a field of adata.obs
    color: Key in adata.obs to color points by.
    use_raw: Use `raw` attribute of `adata` if present.
    """

    adata_raw = __get_raw(adata, use_raw)
    keys = __to_list(keys)
    if color is not None:
        keys.append(color)
    df = __get_df(adata, adata_raw, keys)
    __fix_color_by_data_type(df, color)
    p = hvplot.scatter_matrix(df, c=color, **kwds)
    p.df = df
    return p


def embedding(adata: AnnData, basis: str, keys: Union[None, str, List[str], Tuple[str]] = None,
              cmap: Union[str, List[str], Tuple[str]] = 'viridis',
              alpha: float = 1, size: int = 12,
              width: int = 400, height: int = 400,
              sort: bool = True, cols: int = 2,
              use_raw: bool = None, nbins: int = None, reduce_function: Callable[[np.array], float] = np.mean,
              labels_on_data: bool = False, tooltips: Union[str, List[str], Tuple[str]] = None,
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
    tooltips: List of additional fields to show on hover.
    labels_on_data: Whether to draw labels for categorical features on the plot.
    use_raw: Use `raw` attribute of `adata` if present.
    """

    if keys is None:
        keys = []

    adata_raw = __get_raw(adata, use_raw)
    keys = __to_list(keys)
    if tooltips is None:
        tooltips = []
    tooltips = __to_list(tooltips)
    keywords = dict(fontsize=dict(title=9), padding=0.02, xaxis=False, yaxis=False, nonselection_alpha=0.1,
        tools=['box_select'], cmap=cmap)
    keywords.update(kwds)
    coordinate_columns = ['X_' + basis + c for c in ['1', '2']]
    df = __get_df(adata, adata_raw, keys + tooltips,
        pd.DataFrame(adata.obsm['X_' + basis][:, 0:2], columns=coordinate_columns),
        is_obs=True)
    df_with_coords = df
    if len(keys) == 0:
        keys = ['count']
    plots = []

    if nbins is not None:
        df['count'] = 1.0
        df, df_with_coords = __bin(df, nbins=nbins, coordinate_columns=coordinate_columns,
            reduce_function=reduce_function)

    for key in keys:
        is_color_by_numeric = pd.api.types.is_numeric_dtype(df[key])
        df_to_plot = df
        if sort and is_color_by_numeric:
            df_to_plot = df.sort_values(by=key)
        __create_hover_tool(df, keywords, exclude=coordinate_columns, current=key)
        p = df_to_plot.hvplot.scatter(
            x=coordinate_columns[0],
            y=coordinate_columns[1],
            title=str(key),
            c=key if is_color_by_numeric else None,
            by=key if not is_color_by_numeric else None,
            size=size,
            alpha=alpha,
            colorbar=is_color_by_numeric,
            width=width, height=height, **keywords)
        bounds_stream = __create_bounds_stream(p)
        if not is_color_by_numeric and labels_on_data:
            labels_df = df_to_plot[[coordinate_columns[0], coordinate_columns[1], key]].groupby(key).aggregate(
                np.median)
            labels = hv.Labels({('x', 'y'): labels_df, 'text': labels_df.index.values}, ['x', 'y'], 'text')
            p = p * labels
        p.bounds_stream = bounds_stream
        plots.append(p)

    layout = hv.Layout(plots).cols(cols)
    layout.df = df_with_coords
    return layout


def count_plot(adata: AnnData, by: str, count_by: str, stacked: bool = True, normalize=True, **kwds):
    """
    Generate a composition count plot.

    Parameters:
    adata: Annotated data matrix.
    by: Key for accessing variables of adata.var_names or a field of adata.obs used to group the data.
    by_secondary: Key for accessing variables of adata.var_names or a field of adata.obs used to compute counts within a group.
    reduce_function: Function used to summarize count_by groups
    stacked: Whether bars are stacked.
    normalize: Normalize counts within each group to sum to one.
    """

    adata_raw = __get_raw(adata, False)
    keys = [by, count_by]
    df = __get_df(adata, adata_raw, keys)
    keywords = dict(stacked=stacked, group_label=count_by)
    keywords.update(kwds)
    invert = keywords.get('invert', False)
    if not invert and 'rot' not in keywords:
        keywords['rot'] = 90

    dummy_df = pd.get_dummies(df[count_by])
    df = pd.concat([df, dummy_df], axis=1)
    df = df.groupby(by).agg(np.sum)

    if normalize:
        df = df.T.div(df.sum(axis=1)).T
    p = df.hvplot.bar(by, list(dummy_df.columns.values), **keywords)
    p.df = df
    return p
