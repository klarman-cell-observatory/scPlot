# 0.0.15, January 23, 2020 
- Use same colors as `scanpy`.
- Optionally sort composition_plot 

# 0.0.14, November 20, 2019 
- Use `natsort` to sort categories
- Use colors specified in anndata.uns if available
- Change default binning aggregation method from mean to max
- Added option to link brushing across multiple embeddings

# 0.0.13, October 17, 2019 
- Handle additional data types in composition_plot

# 0.0.12, September 23, 2019 
- Removed statistics from composition_plot
- Handle non-categorical object data types when binning
- Automatically determine the number of columns in embedding plot

# 0.0.11, September 20, 2019 
- Automatically determine point size, changed default categorical color map and updated
legend options for embedding plot

# 0.0.10, August 29, 2019
- Fixed linked brushing in volcano plot

# 0.0.9, August 28, 2019
- Disabled linked brushing in volcano plot as it colors points incorrectly

# 0.0.8, August 28, 2019
- Added volcano plot

# 0.0.7, August 22, 2019
- Renamed count_plot to composition_plot
- Added statistics to composition_plot
- Added variable_feature_plot
- Plots now auto-bin for large datasets

# 0.0.5, August 5, 2019
- Tooltip improvements for embedding plot
- Added count_plot

# 0.0.4, July 25, 2019
- Improved dotplot tooltips

# 0.0.3, July 1, 2019
- Use mode function to summarize categorical data types when binning

# 0.0.2, June 27, 2019
- Return data frame for scatter and embedding in order to get transformed coordinates when binning
- Set default scatter plot color map to viridis

# 0.0.1, June 26, 2019
- First release