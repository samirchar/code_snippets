import matplotlib as plt

def add_top_bar_labels(ax,values):
    rects = ax.patches
    labels = values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height*1.005 , label, ha='center', va='bottom')

