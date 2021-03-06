from sklearn.metrics import precision_score,recall_score, f1_score, confusion_matrix
from scikitplot.metrics import plot_lift_curve,plot_cumulative_gain, plot_precision_recall,plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K

class classification_report:
    
    def __init__(self,y_pred_proba, y_test):
        self.y_pred_proba = y_pred_proba
        self.y_test = y_test
        
    def prediction_prob_binarizer(self, th):
        return (self.y_pred_proba >= th).astype('int')
    
    def report(self, th ,plot_confusion=True, normalize=True, print_=True):
        y_pred = self.prediction_prob_binarizer(th)

        if plot_confusion:
            plot_confusion_matrix(self.y_test, y_pred, normalize=normalize)

        self.cm = confusion_matrix(self.y_test, y_pred)
        self.precision = round(precision_score(self.y_test, y_pred),2)
        self.recall = round(recall_score(self.y_test, y_pred),2)
        self.f1 = round(f1_score(self.y_test, y_pred),2)
        self.fpr = round(self.cm[0,1]/sum(self.cm[0,:]),4)
        
        if print_:
            print('precision = {},  recall = {},  f1 = {} fpr = {} \n\n\n'.format(self.precision,self.recall,self.f1,self.fpr))
   
    def uplift(self, positive_class_proportion, top_n=None, percentile=None, print_=True):

        if percentile:
            n=int(np.round(len(self.y_test)*percentile))
        elif top_n:
            n=top_n
        else:
            print('Must escpeficy top_n or percentile')

        y_true=self.y_test.astype(int)

        temp=pd.DataFrame({'y_pred': self.y_pred_proba, 'y_test': y_true})
        temp=temp.sort_values(by='y_pred',ascending=False).iloc[0:n,:]
        
        positive_in_top_n = sum(temp.y_test)
        response_rate=positive_in_top_n/len(temp.y_test)
        lift=response_rate/positive_class_proportion

        self.lift = lift
        
        
        if print_:
            print('{} positive found in list of {} clients.\n this gives uplift = {}'.format(positive_in_top_n,n,self.lift))
    
    def lift_gain_curves(self):
        
        y_pred_proba_both_classes = np.column_stack([1-self.y_pred_proba,self.y_pred_proba])
        gain = plot_cumulative_gain(self.y_test, y_pred_proba_both_classes, title='Cumulative Gains Curve')
        plt.show()

        lift = plot_lift_curve(self.y_test, y_pred_proba_both_classes, title='Lift curve')
        plt.show()
                

def plot_metrics(history,metric='f1',higher_is_better = True,linestyle = '-',ylim = (0,1)):
    
    try:
        history = history.history
    except:
        pass
    
    metrics = [ i for i in history.keys() if metric in i]
    losses = [ i for i in history.keys() if 'loss' in i]        
    for m in metrics:
        h = np.array(history[m])
        if higher_is_better:
            h = 1 - h
        ax = plt.plot(h ,label = m, color = 'green' if 'val' in m else 'blue',linestyle=linestyle)
    plt.legend()
    ymin,ymax = ylim
    plt.ylim(ymin,ymax)

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val