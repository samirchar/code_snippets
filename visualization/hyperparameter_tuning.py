import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def analyze_hyperparameters(data,score_col,hyperparameters = [],log_scale_vars = []):

    float_hyperparams = list(data.dtypes.index[data.dtypes == 'float'])
    
    if not hyperparameters:
        hyperparameters = data.columns.drop(score_col,axis=1)
        
    for col in hyperparameters:
        if col in float_hyperparams:
            if col in log_scale_vars:
                plt.xscale('log')
            ax = sns.lineplot(x=col,y=score_col,data=data)
            ax.set(title= f"{col} vs {score_col}")
            plt.show()
        else:
            ax = sns.boxenplot(x=col,y=score_col,data=data)
            ax.set(title= f"{col} vs {score_col}")
            plt.show()
    
    
    fig = px.parallel_coordinates(
                              data[hyperparameters+[score_col]],
                              color="f1_val",
                              color_continuous_scale=px.colors.diverging.RdYlGn
                              )
    fig.show()