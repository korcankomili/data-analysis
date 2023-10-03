import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def fetch_csv(data_path): 
    """
        Takes data path \n
        Returns a df
    """
    data = pd.read_csv(data_path)
    return data


def perform_data_checks(df):
    # Null check
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("""
        No missing values found. """)
    else:
        print("""
        Missing values found:""")
        print(null_counts[null_counts > 0])

    # Duplicate check
    if df.duplicated().sum() == 0:
        print("""
        No duplicate rows found. """)
    else:
        print("""
        Duplicate rows found. """)

    # Outlier detection (assuming numerical columns)
    numeric_cols = df.select_dtypes(include=['number'])
    for col in numeric_cols:
        # Check for outliers using a simple threshold
        threshold = 5
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = df[abs(z_scores) > threshold]
        if outliers.empty:
            print(f"""
            No outliers found in column '{col}'. """)
        else:
            print(f"""
            Outliers found in column '{col}': """)


def plot_linear_regressions(df, x_axis, y_axis, ols = True): 
    df = df.copy()  
    if ols == True:
        plot = px.scatter(df,
                          x = x_axis,
                          y = y_axis,
                          trendline = "ols",
                          size = y_axis, 
                          trendline_color_override = 'black',
                        )
    else:
        plot = px.scatter(df,
                          x = x_axis,
                          y = y_axis,
                          size = y_axis, 
                        )
    
    plot.update_layout(title_text=f"Relation of {y_axis} with {x_axis}")
    return plot


def plot_correlation_heatmap(df): 
    """
        Takes a df, \n
        Returns a graph for the correlation matrix of the numeric columns
    """
    fig, ax = plt.subplots(figsize=(12, 6)) 
    
    df = df.copy() 
    corr_matrix = df.select_dtypes(include=['number']).corr()
    
    plot = sns.heatmap(corr_matrix,
                cmap="RdBu",
                annot=True,
                ax=ax)
    
    plot.set_title("Correlation Matrix", fontsize=15)
    
    return plot


def calculate_boxplot_values(df, column_name): 
    """
        Takes df and the column name, \n
        Returns lower_hinge, median, upper_hinge
    """
    column_data = df[column_name]
    
    # Calculate quartiles
    lower_hinge = column_data.quantile(0.25)
    median = column_data.median()
    upper_hinge = column_data.quantile(0.75) 

    IQR = upper_hinge - lower_hinge
    upper_whisker = upper_hinge + 1.5 * IQR

    print(
        f"""
        lower_hinge = {lower_hinge},
        median = {median},
        mean = {df[column_name].mean()}
        upper_hinge = {upper_hinge}
        upper_whisker = {upper_whisker}
        """
    )

    return lower_hinge, median, upper_hinge, upper_whisker


def assign_box_plot_values(df, column_name):
    lower_hinge, median, upper_hinge, upper_whisker = calculate_boxplot_values(df, column_name)

    def assign_value(row):
        if row[column_name] < lower_hinge:
            return "lower_hinge"
        elif row[column_name] < median:
            return "median"
        elif row[column_name] < upper_hinge:
            return "upper_hinge"
        elif row[column_name] < upper_whisker:
            return "upper_whisker"
        else:
            return "high"
    
    new_column_name = column_name + "_category"

    df[new_column_name] = df.apply(assign_value, axis=1)
    return df


def power_regression(df, x_axis, y_axis):
    """
        Takes df, x_axis and y_axis \n
        Returns intercept and slope, also draws the chart
    """
    df = df.copy()  

    x = df[x_axis]
    y = df[y_axis]

    # transform 
    x_transformed = np.log(x.to_numpy())
    y_transformed = np.log(y.to_numpy())

    # fit 
    lr = LinearRegression()
    lr.fit(x_transformed.reshape(-1, 1), y_transformed)
    
    # calculate predicted y values
    x_range = np.linspace(x.min(), x.max(), 100)
    x_range_transformed = np.log(x_range)
    y_pred_transformed = lr.predict(x_range_transformed.reshape(-1, 1))
    y_pred = np.exp(y_pred_transformed)

    # coef and intercept
    slope = lr.coef_[0]
    intercept = np.exp(lr.intercept_) 
    r_squared = lr.score(x_transformed.reshape(-1, 1), y_transformed)

    print(f'Slope: {slope}')
    print(f'Intercept: {intercept}')
    print(f"R2 = % {round(r_squared,4) * 100}") 

    # plot
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f"""{y_axis} = {round(intercept,3)} * ({x_axis} ** {round(slope,3)})""")

    ax.scatter(x, y)
    ax.plot(x_range, y_pred, 'r-')
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    plt.show()
    
    return intercept, slope



def plot_combo_box(df, x_axis, y1, y2, sort_by):
       
    df = df.copy()   
    
    df = df.sort_values(by = sort_by, ascending = True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(y=df[y1], 
               x=df[x_axis], 
               name=y1, 
               orientation="v", 
               marker=dict(color="silver")),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(y=df[y2], 
                   x=df[x_axis], 
                   name=y2, 
                   line=dict(color="blue")),
        secondary_y=True,
    )


    fig.update_layout(
        title=f"{y1} and {y2}",
        xaxis_title = x_axis, 
        yaxis=dict(showgrid=False, zeroline=False, title=y1),
        yaxis2=dict(showgrid=False, zeroline=False, title=y2, side="right", overlaying="y"),
    )

    fig.show()


def plot_histogram(df, bin_by, bin_size, bars_by): 
    """
        Takes df and some parameters \n
        Returns a dynamic histogram

        bin_by: the column your bins will be created from
        bin_size: similar to Tableau, not the bin count but the bin size
        bars_by: the columns your bars will be generated with
    """
    
    df = df.copy()

    bins = pd.cut(df[bin_by], 
                  bins=np.arange(0, df[bin_by].max() + bin_size, bin_size))
    
    df = df.groupby(bins)[bars_by].sum().reset_index()
    
    df['Bins'] = df.index.astype(str)
    
    df['perc_total'] = (df[bars_by] / df[bars_by].sum()* 100) .round(2).astype(str)
    df['perc_total'] = "% " + df['perc_total']
    
    fig = px.bar(df, 
                 x='Bins', 
                 y=bars_by, 
                 text='perc_total', 
                 title=f"{bars_by} by {bin_by} (Bin Size: {bin_size})")
    
    fig.show()