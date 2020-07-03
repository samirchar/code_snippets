import os
import sys
sys.path.append(os.path.abspath(".."))
from tensorflow.keras.layers import (Conv1D,MaxPool1D,Layer,BatchNormalization,Activation,Concatenate)
from tensorflow.nn import relu

class InceptionModule(Layer):
    def __init__(self,num_filters=32,**kwargs):
        super().__init__(**kwargs)
        
        self.num_filters = num_filters

        #Bottleneck convolution
        self.bottleneck = Conv1D(filters=self.num_filters,
                                kernel_size = 1,
                                strides = 1,
                                padding='same')
                                
        self.bn_bottleneck = BatchNormalization()

        self.max_pool = MaxPool1D(pool_size=3,
                                 strides=1,
                                 padding='same')
 
        #Conv, f=10
        self.conv_f10 = Conv1D(filters=self.num_filters,kernel_size = 10,strides = 1,padding='same')
        self.bn_f10 = BatchNormalization()

        #Conv, f=20
        self.conv_f20 = Conv1D(filters=self.num_filters,kernel_size = 20,strides = 1,padding='same')
        self.bn_f20 = BatchNormalization()

        #Conv, f=40
        self.conv_f40 = Conv1D(filters=self.num_filters,kernel_size = 40,strides = 1,padding='same')
        self.bn_f40 = BatchNormalization()

        #Conv, f=1
        self.conv_f1 = Conv1D(filters=self.num_filters,kernel_size = 1, strides = 1,padding='same')
        self.bn_f1 = BatchNormalization()

    '''
    def _default_Conv1D(self,filters,kernel_size):
        return Conv1D(filters=filters,
                    kernel_size = kernel_size,
                    strides = 1,
                    padding='same')

    '''

    def call(self,inputs):

        #Layer 1
        Z_bottleneck = self.bottleneck(inputs)
        Z_bottleneck = self.bn_bottleneck(Z_bottleneck)
        Z_bottleneck = relu(Z_bottleneck)

        Z_maxpool = self.max_pool(inputs)

        #Layer 2
        Z1 = self.conv_f1(Z_maxpool)
        Z1 = self.bn_f1(Z1)
        Z1 = relu(Z1)

        Z2 = self.conv_f10(Z_bottleneck)
        Z2 = self.bn_f10(Z2)
        Z2 = relu(Z2)

        Z3 = self.conv_f20(Z_bottleneck)
        Z3 = self.bn_f20(Z3)
        Z3 = relu(Z3)

        Z4 = self.conv_f40(Z_bottleneck)
        Z4 = self.bn_f40(Z4)
        Z4 = relu(Z4)

        #Layer 3
        Z = Concatenate()([Z1,Z2,Z3,Z4])
        return Z
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_filters':self.num_filters
            })
        return config
    
        