from typing import Union, List, Tuple, Callable, Set

import colorcet
import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from holoviews import dim
from holoviews.plotting.bokeh.callbacks import LinkCallback
from holoviews.plotting.links import Link
from natsort import natsorted
from pandas.api.types import CategoricalDtype


# def sort_by_values(summarized_df):
#     # sort rows by expression
#     sorted_df = summarized_df.sort_values(axis=0, by=list(summarized_df.columns.values), ascending=False)
#     indices = [summarized_df.index.get_loc(c) for c in sorted_df.index]
#     return indices

# Doesn't work for scatter plots colored by categorical variable
class __BrushLinkRange(Link):
    _requires_target = True


class __BrushLinkCallbackRange(LinkCallback):
    source_model = 'selected'
    source_handles = ['cds', 'glyph']
    on_source_changes = ['indices']
    target_model = 'selected'
    target_handles = ['cds', 'glyph']

    source_code = """
       
        target_selected.indices = source_selected.indices;
    """


__BrushLinkRange.register_callback('bokeh', __BrushLinkCallbackRange)


class __BrushLink(Link):
    _requires_target = True


class __BrushLinkCallback(LinkCallback):
    source_model = 'selected'
    source_handles = ['cds']
    on_source_changes = ['indices']
    target_model = 'selected'

    source_code = """
        target_selected.indices = source_selected.indices;
    """


__BrushLink.register_callback('bokeh', __BrushLinkCallback)


def __get_marker_size(count):
    return min(12, (240000.0 if count > 300000 else 120000.0) / count)


def __sort_category(df, by):
    if not pd.api.types.is_categorical_dtype(df[by]):
        df[by] = df[by].astype('category')
    if not df[by].dtype.ordered:
        df[by] = df[by].astype(CategoricalDtype(natsorted(df[by].dtype.categories), ordered=True))


def __auto_bin(df, nbins, width, height):
    if nbins == -1 and df.shape[0] >= 500000:
        nbins = int(max(200, min(width, height) / 2))
    return nbins


def __create_hover_tool(df, keywords: dict, exclude: List, current: str = None, whitelist: List = None):
    """
   Generate hover tool.

   Args:
       keywords: Keyword dict
       exclude: List of columns in df to exclude.
       current: Key in df that is plotted to show 1st in tooltip
   """

    try:
        import bokeh.models
        import holoviews.core.util
        hover_cols = []
        for column in df.columns:
            if column not in exclude and column != current and column not in hover_cols and (
                    whitelist is None or column in whitelist):
                hover_cols.append(column)
        keywords['hover_cols'] = hover_cols
        tooltips = []
        if current is not None:
            tooltips.append((current, '@{' + holoviews.core.util.dimension_sanitizer(current) + '}'))
        for hover_col in hover_cols:
            tooltips.append((hover_col, '@{' + holoviews.core.util.dimension_sanitizer(hover_col) + '}'))
        tools = keywords.get('tools', [])
        keywords['tools'] = tools + [bokeh.models.HoverTool(tooltips=tooltips)]
    except ModuleNotFoundError:
        pass


scanpy_default_102 = ['#FFFF00', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059', '#FFDBE5', '#7A4900',
                      '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87', '#5A0007', '#809693', '#6A3A4C',
                      '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80', '#61615A', '#BA0900', '#6B7900', '#00C2A0',
                      '#FFAA92', '#FF90C9', '#B903AA', '#D16100', '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018',
                      '#0AA6D8', '#013349', '#00846F', '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744', '#C0B9B2',
                      '#C2FF99', '#001E09', '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1',
                      '#788D66', '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648', '#0086ED', '#886F4C',
                      '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81', '#575329',
                      '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00', '#7900D7', '#A77500',
                      '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF', '#9B9700', '#549E79', '#FFF69F', '#201625',
                      '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#922329', '#5B4534', '#FDE8DC', '#404E55', '#0089A3',
                      '#CB7E98', '#A4E804', '#324E72']
scanpy_default_20 = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2', '#b5bd61',
                     '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2',
                     '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31']
scanpy_default_28 = ['#023fa5', '#7d87b9', '#bec1d4', '#d6bcc0', '#bb7784', '#8e063b', '#4a6fe3', '#8595e1',
                     '#b5bbe3', '#e6afb9', '#e07b91', '#d33f6a', '#11c638', '#8dd593', '#c6dec7', '#ead3c6',
                     '#f0b98d', '#ef9708', '#0fcfc0', '#9cded6', '#d5eae7', '#f3e1eb', '#f6c4e1', '#f79cd4',
                     '#7f7f7f', '#c7c7c7', '#1CE6FF', '#336600']


def __get_scanpy_colors(df, value_to_plot):
    from matplotlib import rcParams
    categories = df[value_to_plot].cat.categories
    length = len(categories)

    # check if default matplotlib palette has enough colors
    if len(rcParams['axes.prop_cycle'].by_key()['color']) >= length:
        cc = rcParams['axes.prop_cycle']()
        palette = [next(cc)['color'] for _ in range(length)]

    else:
        if length <= 20:
            palette = scanpy_default_20
        elif length <= 28:
            palette = scanpy_default_28
        elif length <= len(scanpy_default_102):  # 103 colors
            palette = scanpy_default_102
        else:
            palette = ['grey' for _ in range(length)]
    return palette


def __set_colors_scanpy(adata, value_to_plot):
    palette = __get_scanpy_colors(adata.obs[value_to_plot], value_to_plot)
    colors = palette[:len(adata.obs[value_to_plot].cat.categories)]
    adata.uns[value_to_plot + '_colors'] = colors
    return colors


def __fix_cmap(df, key, palette=None):
    # check for missing values in palette
    palette = palette.copy()
    colors = colorcet.b_glasbey_category10
    ncolors = len(colors)
    i = 0
    for c in df[key].cat.categories:
        if c not in palette:
            palette[c] = colors[i % ncolors]
            i += 1
    return palette


def __get_category_cmap(adata, key):
    if key is not None:
        color_key = f"{key}_colors"
        colors = None
        if color_key in adata.uns:
            adata_categories = adata.obs[key].cat.categories
            if len(adata.uns[color_key]) == len(adata_categories):
                colors = adata.uns[color_key]
        if colors is None:
            colors = __get_scanpy_colors(adata.obs, key)
        ncolors = len(colors)
        color_map = {}
        adata_categories = adata.obs[key].cat.categories
        for i in range(len(adata_categories)):
            color_map[adata_categories[i]] = colors[i % ncolors]
        return color_map


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
    size_ticks_pixels = np.interp(size_ticks, (size_min, size_max), (dot_min, dot_max))
    size_tick_labels = [size_tick_labels_format.format(x) for x in size_ticks]
    points = hv.Points(
        {'x': np.repeat(0.15, len(size_ticks)), 'y': np.arange(len(size_ticks), 0, -1),
         'size': size_ticks_pixels},
        vdims='size').opts(xaxis=None, color='black', yaxis=None, size=dim('size'))
    labels = hv.Labels(
        {'x': np.repeat(0.3, len(size_ticks)), 'y': np.arange(len(size_ticks), 0, -1),
         'text': size_tick_labels},
        ['x', 'y'], 'text').opts(text_align='left', text_font_size='9pt')
    overlay = (points * labels)
    overlay.opts(width=dot_max + 100, height=int(len(size_ticks) * (dot_max + 12)), xlim=(0, 1),
        ylim=(0, len(size_ticks) + 1),
        invert_yaxis=True, shared_axes=False, show_frame=False)
    return overlay


def __fix_color_by_data_type(df, by):
    if by is not None and pd.api.types.is_bool_dtype(df[by]):
        df[by] = df[by].astype(str).astype('category')  # hvplot does not handle boolean type for colors


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
    for i in range(len(keys)):
        key = keys[i]
        if df is None:
            if isinstance(key, np.ndarray):
                is_obs = len(key) == adata.shape[0]
            else:
                is_obs = key not in adata.var
            df = pd.DataFrame(data=dict(id=(adata.obs.index.values if is_obs else adata.var.index.values)))
        if isinstance(key, np.ndarray):
            values = key
            key = str(i)
            keys[i] = key
        elif key in adata_raw.var_names and is_obs:
            X = adata_raw.obs_vector(key)
            #  X = adata_raw[:, key].X
            if scipy.sparse.issparse(X):
                X = X.toarray()
            values = X
        elif key in adata.obs and is_obs:
            values = adata.obs[key].values
        elif key in adata.var and not is_obs:
            values = adata.var[key].values
        else:
            raise ValueError('{} not found'.format(key))
        df[key] = values
    return df


def mode_and_purity(x):
    value_counts = x.value_counts(sort=False)
    largest = value_counts.nlargest(1)
    purity = largest[0] / value_counts.sum()
    return largest.index[0], purity


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
        if column not in coordinate_columns:
            if column == 'count':
                agg_func[column] = 'sum'
            elif pd.api.types.is_numeric_dtype(df[column]):
                agg_func[column] = reduce_function
            else:  # pd.api.types.is_categorical_dtype(df[column]):
                agg_func[column] = mode_and_purity
    return df.groupby(coordinate_columns, as_index=False).agg(agg_func), df[coordinate_columns]


def violin(adata: AnnData, keys: Union[str, List[str], Tuple[str]], by: str = None,
           width: int = 300, cmap: Union[str, List[str], Tuple[str]] = None, cols: int = None,
           use_raw: bool = None, **kwds) -> hv.core.element.Element:
    """
    Generate a violin plot.

    Args:
        adata: Annotated data matrix.
        keys: Keys for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
        by: Group plot by specified observation.
        width: Plot width.
        cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
        cols: Number of columns for laying out multiple plots
        use_raw: Use `raw` attribute of `adata` if present.
    """
    if cols is None:
        cols = 3
    adata_raw = __get_raw(adata, use_raw)
    keys = __to_list(keys)
    df = __get_df(adata, adata_raw, keys + ([] if by is None else [by]))
    cmap = __get_category_cmap(adata, by) if cmap is None else __fix_cmap(df, by, cmap)
    plots = []
    keywords = dict(padding=0.02, cmap=cmap, rot=90)
    keywords.update(kwds)

    __fix_color_by_data_type(df, by)
    if by is not None:
        __sort_category(df, by)
    for key in keys:
        p = df.hvplot.violin(key, width=width, by=by, violin_color=by, **keywords)
        plots.append(p)

    return hv.Layout(plots).cols(cols)


def heatmap(adata: AnnData, keys: Union[str, List[str], Tuple[str]], by: str,
            reduce_function: Callable[[np.ndarray], float] = np.mean,
            use_raw: bool = None, cmap: Union[str, List[str], Tuple[str]] = 'Reds', **kwds) -> hv.core.element.Element:
    """
    Generate a heatmap.

    Args:
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
        X = adata_raw.obs_vector(key)
        # X = adata_raw[:, key].X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        _df = pd.DataFrame(X, columns=['value'])
        _df['feature'] = key
        _df[by] = adata.obs[by].values
        df = _df if df is None else pd.concat((df, _df))

    __sort_category(df, by)
    df['feature'] = df['feature'].astype(CategoricalDtype(keys, ordered=True))
    return df.hvplot.heatmap(x='feature', y=by, C='value', reduce_function=reduce_function, **keywords)


def scatter(adata: AnnData, x: str, y: str, color: str = None, size: Union[int, str] = None,
            dot_min=2, dot_max=14, use_raw: bool = None, sort: bool = True, width: int = 400, height: int = 400,
            nbins: int = -1, reduce_function: Callable[[np.array], float] = np.max,
            cmap: Union[str, List[str], Tuple[str]] = None, palette: Union[str, List[str], Tuple[str]] = None,
            **kwds) -> hv.core.element.Element:
    """
    Generate a scatter plot.

    Args:
        adata: Annotated data matrix.
        x: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
        y: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
        cmap: Color map for continuous variables.
        palette: Color map for categorical variables.
        color: Field in .var_names, adata.var, or adata.obs to color the points by.
        sort: Plot higher color by values on top of lower values.
        width: Chart width.
        height: Chart height.
        size: Field in .var_names, adata.var, or adata.obs to size the points by or a pixel size.
        dot_min: Minimum dot size when sizing points by a field.
        dot_max: Maximum dot size when sizing points by a field.
        use_raw: Use `raw` attribute of `adata` if present.
        nbins: Number of bins used to summarize plot on a grid. Useful for large datasets. Negative one means automatically bin the plot.
        reduce_function: Function used to summarize overlapping cells if nbins is specified
    """
    return __scatter(adata=adata, x=x, y=y, color=color, size=size, dot_min=dot_min, dot_max=dot_max, use_raw=use_raw,
        sort=sort, width=width, height=height, nbins=nbins, reduce_function=reduce_function, cmap=cmap, palette=palette,
        is_scatter=True, **kwds)


def line(adata: AnnData, x: str, y: str,
         use_raw: bool = None, width: int = 400, height: int = 400,
         nbins: int = None, reduce_function: Callable[[np.array], float] = np.max, **kwds) -> hv.core.element.Element:
    """
    Generate a scatter plot.

    Args:
        adata: Annotated data matrix.
        x: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
        y: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
        use_raw: Use `raw` attribute of `adata` if present.
        width: Chart width.
        height: Chart height.
        nbins: Number of bins used to summarize plot on a grid. Useful for large datasets.
        reduce_function: Function used to summarize overlapping cells if nbins is specified
    """
    return __scatter(adata=adata, x=x, y=y, use_raw=use_raw, sort=False, width=width, height=height, nbins=nbins,
        reduce_function=reduce_function, is_scatter=False,
        **kwds)


def __scatter(adata: AnnData, x: str, y: str, color=None, size: Union[int, str] = None,
              dot_min=2, dot_max=14, use_raw: bool = None, sort: bool = True, width: int = 400, height: int = 400,
              nbins: int = None, reduce_function: Callable[[np.array], float] = np.max,
              cmap: Union[str, List[str], Tuple[str]] = None, palette: Union[str, List[str], Tuple[str]] = None,
              is_scatter=True, **kwds) -> hv.core.element.Element:
    """
    Generate a scatter plot.

    Args:
        adata: Annotated data matrix.
        x: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
        y: Key for accessing variables of adata.var_names, field of adata.var, or field of adata.obs
        cmap: Color map for continous variables.
        palette: Color map for categorical variables.
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
    is_size_by = size is not None and is_scatter
    is_color_by = color is not None and is_scatter
    keywords = dict(fontsize=dict(title=9), nonselection_alpha=0.1, padding=0.02, xaxis=True, yaxis=True, width=width,
        height=height, alpha=1, tools=['box_select'])
    keywords.update(kwds)
    keys = [x, y]
    if is_color_by:
        keys.append(color)
    if is_size_by:
        keys.append(size)

    df = __get_df(adata, adata_raw, keys)

    # keys might have been modified by __get_df if key was an array instead of a string
    x = keys[0]
    y = keys[1]
    if is_color_by:
        color = keys[2]
    if is_size_by:
        size = keys[3 if is_color_by else 2]

    nbins = __auto_bin(df, nbins, width, height)
    bin_data = nbins is not None and nbins > 0
    df_with_coords = df
    hover_cols = keywords.get('hover_cols', [])
    if bin_data:
        df['count'] = 1.0
        hover_cols.append('count')
        df, df_with_coords = __bin(df, nbins=nbins, coordinate_columns=[x, y], reduce_function=reduce_function)
    else:
        hover_cols.append('id')
    keywords['hover_cols'] = hover_cols

    if is_color_by:
        is_color_by_numeric = False
        is_categorical = bin_data and pd.api.types.is_object_dtype(df[color])

        if is_categorical:
            df = pd.DataFrame(df[color].tolist(), columns=[color, str(color) + '_purity']).join(df,
                rsuffix='orig_')
        else:
            is_color_by_numeric = pd.api.types.is_numeric_dtype(df[color])
            if not is_color_by_numeric:
                __fix_color_by_data_type(df, color)

        __fix_scatter_colors(adata, df, color, is_color_by_numeric, cmap, palette, keywords)
        use_c = is_color_by_numeric
        keywords['c' if use_c else 'by'] = color
        if is_color_by_numeric:
            keywords.update(dict(colorbar=True))
            if sort:
                df = df.sort_values(by=color)

    if is_size_by:
        size_min = df[size].min()
        size_max = df[size].max()
        size_pixels = np.interp(df[size], (size_min, size_max), (dot_min, dot_max))
        df['pixels'] = size_pixels
        keywords['s'] = 'pixels'
        hover_cols = keywords.get('hover_cols', [])
        hover_cols.append(size)
        keywords['hover_cols'] = hover_cols
    if is_scatter:
        p = df.hvplot.scatter(x=x, y=y, **keywords)
    else:  # line plot
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


def dotplot(adata: AnnData, keys: Union[str, List[str], Tuple[str]], by: str,
            reduce_function: Callable[[np.ndarray], float] = np.mean,
            fraction_min: float = 0, fraction_max: float = None, dot_min: int = 1, dot_max: int = 26,
            use_raw: bool = None, cmap: Union[str, List[str], Tuple[str]] = 'Reds',
            sort_function: Callable[[pd.DataFrame], List[str]] = None, **kwds) -> hv.core.element.Element:
    """
    Generate a dot plot.

    Args:
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
        sort_function: Optional function that accepts summarized data frame and returns a list of row indices in the order to render in the heatmap.
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

    summarized_df = df.groupby(by).aggregate([reduce_function, non_zero])
    if sort_function is not None:  # sort categories
        row_indices = sort_function(summarized_df)
        summarized_df = summarized_df.iloc[row_indices]
    else:
        summarized_df = summarized_df.loc[natsorted(summarized_df.index, reverse=True)]

    mean_columns = []
    frac_columns = []
    for i in range(len(summarized_df.columns)):
        if i % 2 == 0:
            mean_columns.append(summarized_df.columns[i])
        else:
            frac_columns.append(summarized_df.columns[i])
    # features on columns, by on rows
    fraction_df = summarized_df[frac_columns]
    mean_df = summarized_df[mean_columns]

    y, x = np.indices(mean_df.shape)
    y = y.flatten()
    x = x.flatten()
    fraction = fraction_df.values.flatten()
    if fraction_max is None:
        fraction_max = fraction.max()
    pixels = np.interp(fraction, (fraction_min, fraction_max), (dot_min, dot_max))
    pixels = pixels * pixels  # hvplot takes the sqrt of size
    summary_values = mean_df.values.flatten()
    xlabel = [keys[i] for i in range(len(keys))]
    ylabel = [str(summarized_df.index[i]) for i in range(len(summarized_df.index))]
    dotplot_df = pd.DataFrame(
        data=dict(x=x, y=y, value=summary_values, pixels=pixels, fraction=fraction, xlabel=np.array(xlabel)[x],
            ylabel=np.array(ylabel)[y]))

    xticks = [(i, keys[i]) for i in range(len(keys))]
    yticks = [(i, str(summarized_df.index[i])) for i in range(len(summarized_df.index))]

    # note we take the max label string length as an approximation of width of labels in pixels
    keywords['width'] = int(
        np.ceil(((dot_max + 1) + 4) * len(xticks) + dotplot_df['ylabel'].str.len().max()) + dot_max + 100)
    keywords['height'] = int(np.ceil(((dot_max + 1) + 4) * len(yticks) + dotplot_df['xlabel'].str.len().max()) + 50)
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

    p = dotplot_df.hvplot.scatter(x='x', y='y', xlim=(-1, len(xticks)), ylim=(-1, len(yticks)),
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


def scatter_matrix(adata: AnnData, keys: Union[str, List[str], Tuple[str]], color=None, use_raw: bool = None,
                   **kwds) -> hv.core.element.Element:
    """
    Generate a scatter plot matrix.

    Args:
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


def __fix_scatter_colors(adata, df_to_plot, key, is_color_by_numeric, cmap, palette, keywords):
    color_keyword_keep = 'cmap'  # dict of colors to use with 'c'
    color_keyword_delete = 'color'  # key in df containing colors to use with 'by'
    use_c = is_color_by_numeric
    # 'c' does not show clickable legend, but respects dict color map, 'by' shows legend but does not respect color map, use array of colors

    if is_color_by_numeric:
        color_map = 'viridis' if cmap is None else cmap
    else:
        __sort_category(df_to_plot, key)  # for legend
        color_map = __get_category_cmap(adata, key) if palette is None else __fix_cmap(df_to_plot, key, palette)
        df_to_plot['__color'] = df_to_plot[key].apply(lambda x: color_map[x])
        color_map = '__color'
        color_keyword_keep = 'color'
        color_keyword_delete = 'cmap'

    keywords[color_keyword_keep] = color_map
    if color_keyword_delete in keywords:
        del keywords[color_keyword_delete]


def embedding(adata: AnnData, basis: Union[str, List[str], Tuple[str]],
              keys: Union[None, str, List[str], Tuple[str]] = None,
              cmap: Union[str, List[str], Tuple[str]] = None, palette: Union[str, List[str], Tuple[str]] = None,
              alpha: float = 1, size: float = None, width: int = 400, height: int = 400, sort: bool = True,
              cols: int = None, use_raw: bool = None, nbins: int = -1,
              reduce_function: Callable[[np.array], float] = np.max,
              brush_categorical: bool = False, legend: str = 'right',
              tooltips: Union[str, List[str], Tuple[str]] = None,
              legend_font_size: Union[int, str] = None, opacity_min: float = 0, opacity_max: float = 1,
              **kwds) -> hv.core.element.Element:
    """
    Generate an embedding plot.

    Args:
        adata: Annotated data matrix.
        keys: Key for accessing variables of adata.var_names or a field of adata.obs used to color the plot. Can also use `count` to plot cell count when binning.
        basis: String in adata.obsm containing coordinates.
        alpha: Points alpha value.
        size: Point pixel size.
        sort: Plot higher values on top of lower values. Disable for linked brushing.
        brush_categorical: Enable linked brushing on categorical variables (disables categorical legend).
        cmap: Color map for continous variables.
        palette: Color map for categorical variables.
        nbins: Number of bins used to summarize plot on a grid. Useful for large datasets. Negative one means automatically bin the plot.
        reduce_function: Function used to summarize overlapping cells if nbins is specified.
        cols: Number of columns for laying out multiple plots
        width: Plot width.
        height: Plot height.
        tooltips: List of additional fields to show on hover.
        legend: `top', 'bottom', 'left', 'right', or 'data' to draw labels for categorical features on the plot.
        legend_font_size: Font size for `labels_on_data`
        use_raw: Use `raw` attribute of `adata` if present.
        opacity_min: Minimum value for encoding categorical data purity when binning using opacity.
        opacity_max: Maximum value for encoding categorical data purity when binning using opacity.
    """

    if keys is None:
        keys = []
    basis = __to_list(basis)
    adata_raw = __get_raw(adata, use_raw)
    keys = __to_list(keys)
    if tooltips is None:
        tooltips = []
    tooltips = __to_list(tooltips)
    if legend_font_size is None:
        legend_font_size = '12pt'
    labels_on_data = legend == 'data'
    keywords = dict(fontsize=dict(title=9, legend=legend_font_size), padding=0.02 if not labels_on_data else 0.05,
        xaxis=False, yaxis=False,
        nonselection_alpha=0.1,
        tools=['box_select'], legend=not legend == 'data')
    plots = []
    charts_to_brush = []
    keywords.update(kwds)
    data_df = __get_df(adata, adata_raw, keys + tooltips, is_obs=True)
    density = len(keys) == 0
    if density:
        keys = ['count']

    for b in basis:
        df = data_df.copy()
        coordinate_columns = ['X_' + b + c for c in ['1', '2']]
        df = pd.concat((df, pd.DataFrame(adata.obsm['X_' + b][:, 0:2], columns=coordinate_columns)), axis=1)
        nbins = __auto_bin(df, nbins, width, height)
        df_with_coords = df
        bin_data = nbins is not None and nbins > 0

        if bin_data or density:
            df['count'] = 1.0
        if bin_data:
            df, df_with_coords = __bin(df, nbins=nbins, coordinate_columns=coordinate_columns,
                reduce_function=reduce_function)

        if size is None:
            size = __get_marker_size(df.shape[0])

        for key in keys:
            is_color_by_numeric = False
            is_categorical_binned = bin_data and pd.api.types.is_object_dtype(df[key])
            if is_categorical_binned:
                df_to_plot = pd.DataFrame(df[key].tolist(), columns=[key, str(key) + '_purity']).join(df,
                    rsuffix='orig_')
            else:
                is_color_by_numeric = pd.api.types.is_numeric_dtype(df[key])
                if not is_color_by_numeric:
                    __fix_color_by_data_type(df, key)
                df_to_plot = df
            if sort and is_color_by_numeric:
                df_to_plot = df.sort_values(by=key)
            __create_hover_tool(df_to_plot, keywords, exclude=coordinate_columns, current=key)
            use_c = is_color_by_numeric or brush_categorical
            __fix_scatter_colors(adata, df_to_plot, key, is_color_by_numeric, cmap, palette, keywords)

            if is_categorical_binned:
                point_opacity = np.interp(df_to_plot[str(key) + '_purity'],
                    (df_to_plot[str(key) + '_purity'].min(), df_to_plot[str(key) + '_purity'].max()),
                    (opacity_min, opacity_max))
                df_to_plot['__point_opacity'] = point_opacity
            p = df_to_plot.hvplot.scatter(
                x=coordinate_columns[0],
                y=coordinate_columns[1],
                title=str(key),
                c=key if use_c else None,
                by=key if not use_c else None,
                size=size,
                alpha='__point_opacity' if is_categorical_binned else alpha,
                colorbar=is_color_by_numeric,
                width=width,
                height=height,
                **keywords)
            bounds_stream = __create_bounds_stream(p)
            if not sort and not bin_data:
                charts_to_brush.append(p)
            if not is_color_by_numeric and labels_on_data:
                labels_df = df_to_plot[[coordinate_columns[0], coordinate_columns[1], key]].groupby(key).aggregate(
                    np.median)
                labels = hv.Labels({('x', 'y'): labels_df, 'text': labels_df.index.values}, ['x', 'y'], 'text').opts(
                    text_font_size=legend_font_size)
                p = p * labels
            p.bounds_stream = bounds_stream
            plots.append(p)
    for i in range(len(charts_to_brush)):
        for j in range(i):
            __BrushLinkRange(charts_to_brush[i], charts_to_brush[j])
            __BrushLinkRange(charts_to_brush[j], charts_to_brush[i])
    if cols is None:
        cols = 1 if width > 500 else 2
    layout = hv.Layout(plots).cols(cols)
    layout.df = df_with_coords
    return layout


def variable_feature_plot(adata: AnnData, **kwds) -> hv.core.element.Element:
    """
    Generate a variable feature plot.

    Args:
        adata: Annotated data matrix.
   """

    if 'hvf_loess' in adata.var:
        keywords = dict(x='mean', y='var', y_fit='hvf_loess', color='highly_variable_features',
            xlabel='Mean log expression', ylabel='Variance of log expression')
    else:
        keywords = dict(x='means', y='dispersions_norm', y_fit=None, color='highly_variable',
            xlabel='Mean log expression', ylabel='Normalized dispersion')

    keywords.update(kwds)
    x = keywords.pop('x')
    y = keywords.pop('y')
    color = keywords.pop('color')
    xlabel = keywords.pop('xlabel')
    ylabel = keywords.pop('ylabel')
    y_fit = keywords.pop('y_fit')
    line_color = keywords.pop('line_color', 'black')

    if y_fit is not None and y_fit in adata.var:
        return scatter(adata, x=x, y=y, xlabel=xlabel, color=color,
            ylabel=ylabel, **keywords) * line(adata, x=x, y=y_fit, line_color=line_color)
    else:
        return scatter(adata, x=x, y=y, color=color, xlabel=xlabel, ylabel=ylabel)


def volcano(adata: AnnData, basis: str = 'de_res', x: str = 'log_fold_change', y: str = 't_qval',
            x_cutoff: float = 1, y_cutoff: float = 0.05, cluster_ids: Union[List, Tuple, Set] = None,
            **kwds) -> hv.core.element.Element:
    """
    Generate a volcano plot.

    Args:
        adata: Annotated data matrix.
        basis: String in adata.varm containing statistics to plot.
        x: Field in basis to plot on x-axis. Field is assumed to end with :cluster_id (e.g. log_fold_change:1).
        y: Field in basis to plot on y-axis. Field is assumed to end with :cluster_id (e.g. t_qval:1)..
        x_cutoff: Highlight items >= x_cutoff or <=-x_cutoff
        y_cutoff: Highlight items >= y_cutoff
        cluster_ids: Optional list of cluster ids to include. If unspecified, plots are shown for all clusters.
   """
    de_results = adata.varm[basis]
    names = de_results.dtype.names  # stat:cluster e.g. 'mwu_pval:13'
    cluster_to_xy = {}

    keywords = dict(fontsize=dict(title=9), nonselection_line_color=None, line_color='black',
        selection_line_color='black', line_width=0.3, nonselection_alpha=0.05,
        padding=0.02, xaxis=True, yaxis=True, alpha=0.9,
        tools=['box_select'], hover_cols=['id'],
        cmap={'Up': '#e41a1c', 'Down': '#377eb8', 'Not significant': '#bdbdbd'})
    keywords.update(kwds)
    for name in names:
        xy_index = -1
        if name.startswith(x):
            xy_index = 0
        elif name.startswith(y):
            xy_index = 1
        if xy_index != -1:
            cluster_id = name[name.rindex(':') + 1:]
            if cluster_ids is None or (cluster_ids is not None and cluster_id in cluster_ids):
                xy = cluster_to_xy.get(cluster_id, None)
                if xy is None:
                    xy = [None, None]
                    cluster_to_xy[cluster_id] = xy
                xy[xy_index] = name
    plots = []
    cluster_ids = cluster_to_xy.keys()
    df = pd.DataFrame(dict(id=adata.var.index.values))
    filtered_cluster_ids = []
    for cluster_id in cluster_ids:
        xy = cluster_to_xy[cluster_id]
        if xy[0] is not None and xy[1] is not None:
            filtered_cluster_ids.append(cluster_id)
            x_column = '{}_{}'.format(x, cluster_id)
            y_column = '{}_{}'.format(y, cluster_id)
            y_log_column = '{}_{}_log'.format(y, cluster_id)
            status_column = '{}_status'.format(cluster_id)
            df[x_column] = de_results[xy[0]]
            df[y_column] = de_results[xy[1]]
            df[status_column] = 'Not significant'
            df.loc[(df[y_column] <= y_cutoff) & (df[x_column] >= x_cutoff), status_column] = 'Up'
            df.loc[(df[y_column] <= y_cutoff) & (df[x_column] < -x_cutoff), status_column] = 'Down'
            df[y_log_column] = -np.log10(df[y_column] + 1e-12)
    for cluster_id in filtered_cluster_ids:
        x_column = '{}_{}'.format(x, cluster_id)
        y_column = '{}_{}'.format(y, cluster_id)
        y_log_column = '{}_{}_log'.format(y, cluster_id)
        status_column = '{}_status'.format(cluster_id)
        __create_hover_tool(df, keywords, exclude=[], whitelist=['id', x_column, y_column])
        p = df.hvplot.scatter(x=x_column, y=y_log_column, title=str(
            cluster_id), c=status_column, xlabel=str(x), ylabel='-log10 ' + str(y), **keywords)
        plots.append(p)
    # shared_datasource for linked brushing colors points incorrectly
    for i in range(len(plots)):
        for j in range(i):
            __BrushLink(plots[i], plots[j])
            __BrushLink(plots[j], plots[i])
    result = hv.Layout(plots).cols(1)
    result.df = df
    return result


def composition_plot(adata: AnnData, by: str, condition: str, stacked: bool = True, normalize: bool = True,
                     condition_sort_by: str = None, cmap: Union[str, List[str], Tuple[str]] = None,
                     **kwds) -> hv.core.element.Element:
    """
     Generate a composition plot, which shows the percentage of observations from every condition within each cluster (by).

     Args:
         adata: Annotated data matrix.
         by: Key for accessing variables of adata.var_names or a field of adata.obs used to group the data.
         condition: Key for accessing variables of adata.var_names or a field of adata.obs used to compute counts within a group.
         stacked: Whether bars are stacked.
         normalize: Normalize counts within each group to sum to one.
         condition_sort_by: Sort condition within each group by max, mean, natsorted, or None.
         cmap: Color map name (hv.plotting.list_cmaps()) or a list of hex colors. See http://holoviews.org/user_guide/Styling_Plots.html for more information.
     """

    adata_raw = __get_raw(adata, False)
    keys = [by, condition]
    adata_df = __get_df(adata, adata_raw, keys)

    for column in adata_df:
        if not pd.api.types.is_categorical_dtype(adata_df[column]):
            adata_df[column] = adata_df[column].astype(str).astype('category')

    cmap = __get_category_cmap(adata_df, condition) if cmap is None else __fix_cmap(adata_df, condition, cmap)
    keywords = dict(stacked=stacked, group_label=condition)
    keywords.update(kwds)
    invert = keywords.get('invert', False)
    if not invert and 'rot' not in keywords:
        keywords['rot'] = 90

    dummy_df = pd.get_dummies(adata_df[condition])
    df = pd.concat([adata_df, dummy_df], axis=1)
    df = df.groupby(by).agg(np.sum)

    if normalize:
        df = df.T.div(df.sum(axis=1)).T

    if not (pd.api.types.is_categorical_dtype(df.index) and df.index.dtype.ordered):
        df = df.loc[natsorted(df.index)]

    secondary = dummy_df.columns.values

    if condition_sort_by == 'max' or condition_sort_by == 'mean':
        secondary_sort = df.values.max(axis=0) if condition_sort_by == 'max' else df.values.mean(axis=0)
        index = np.flip(np.argsort(secondary_sort))
        secondary = secondary[index]
    elif condition_sort_by == 'natsorted':
        secondary = natsorted(secondary)
    secondary = list(secondary)
    p = df.hvplot.bar(by, secondary, cmap=cmap, **keywords)
    p.df = df
    return p
