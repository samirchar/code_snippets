import os
import sys
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll.stochastic import sample
from functools import partial
from glob import glob
import re


from src.code_snippets.evaluation.model_evaluation import plot_metrics, f1_metric
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    Conv1D,
    Dropout,
    Dense,
    BatchNormalization,
    Activation,
    Bidirectional,
    SpatialDropout1D,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.backend import clear_session
from src.code_snippets.utils.abstract_classes import Trainer
from src.code_snippets.dataprep.embeddings_preprocessing.glove.reader import (
    read_glove_file,
    get_word_index_dicts,
)
from src.code_snippets.dataprep.embeddings_preprocessing.data_preparation import (
    pretrained_embedding_layer,
)
from src.code_snippets.models.hyperparameter_tuning import safeHyperopt


class BidirectionalLSTM(Trainer):
    def __init__(self, train_data, val_data, embedding_dir, model=None):

        self.train_data = train_data
        self.val_data = val_data
        self.embedding_dir = embedding_dir
        self.gensim_model = read_glove_file(self.embedding_dir)
        self.word_to_index, self.index_to_words = get_word_index_dicts(
            self.gensim_model
        )

        self.m_X, self.n_X = self.train_data["X_indices"].shape
        self.m_X_aux, self.n_X_aux = self.train_data["X_aux"].shape

        self.model = model
        self.model_name = type(self).__name__
    def set_model(
        self,
        n_units=2 ** 7,
        add_recurrent_layer=False,
        dropout=0.1,
        spatial_dropout=0,
        hidden_dense_units=0,
        learning_rate=0.01,
        bidirectional=True,
        global_max_pool=False,
        global_avg_pool=False,
        seed=100,
    ):

        # Input layer
        sentence_indices = Input((self.n_X), dtype="int32")

        # Embedding Layer
        embedding_layer = pretrained_embedding_layer(
            self.gensim_model, self.word_to_index
        )

        # Propagate sentence_indices through your embedding layer
        embeddings = embedding_layer(sentence_indices)
        embeddings = SpatialDropout1D(spatial_dropout, seed=seed)(embeddings)
        # LSTM 1
        if bidirectional:
            X = Bidirectional(
                LSTM(n_units, return_sequences=True if add_recurrent_layer else False)
            )(embeddings)
        else:
            X = LSTM(n_units, return_sequences=True if add_recurrent_layer else False)(
                embeddings
            )

        if add_recurrent_layer:
            # LSTM 2
            return_seq = True if (global_avg_pool | global_max_pool) else False

            if bidirectional:
                if return_seq:

                    items_to_concat = []

                    (
                        lstm,
                        forward_h,
                        forward_c,
                        backward_h,
                        backward_c,
                    ) = Bidirectional(
                        LSTM(n_units, return_sequences=True, return_state=True)
                    )(
                        X
                    )

                    state_h = concatenate([forward_h, backward_h])
                    state_c = concatenate([forward_c, backward_c])

                    items_to_concat.append(state_h)

                    if global_avg_pool:
                        avg_pool = GlobalAveragePooling1D()(lstm)
                        items_to_concat.append(avg_pool)

                    if global_max_pool:
                        max_pool = GlobalMaxPooling1D()(lstm)
                        items_to_concat.append(max_pool)

                    X = concatenate(items_to_concat)

                else:
                    X = Bidirectional(LSTM(n_units, return_sequences=False))(X)
            else:
                X = LSTM(n_units, return_sequences=False)(X)

        if hidden_dense_units > 0:
            # Hiden Dense Layer 1
            X = Dense(hidden_dense_units)(X)
            X = BatchNormalization()(X)
            X = Activation(activation="relu")(X)
            X = Dropout(dropout, seed=seed)(X)

        # Output layer
        X = Dense(1, activation="sigmoid")(X)

        self.model = Model(sentence_indices, X)

        opt = Adam(learning_rate=learning_rate)
        self.model.compile(
            loss="binary_crossentropy", metrics=[f1_metric], optimizer=opt
        )

    def save_model(self,file_path):
        self.model.save(file_path)

    def fit_model(
        self,
        epochs,
        batch_size,
        use_early_stopping=True,
        monitor="val_f1_metric",
        patience=15,
        min_delta=1e-4,
    ):
        self.early_stopping = EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        )

        self.history = self.model.fit(
            self.train_data["X_indices"],
            self.train_data["y"],
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=[self.val_data["X_indices"], self.val_data["y"]],
            verbose=1,
            callbacks=[self.early_stopping] if use_early_stopping else None,
        )

    def generate_metrics(self, X_test, y_test, metric="f1_score"):

        y_test = y_test.astype("float32")
        y_pred = self.model.predict(X_test)

        if metric == "f1_score":
            score = f1_metric(y_test, y_pred).numpy()

        return score

    def hyperopt_model(self, params: dict, verbose: int = 0):
        # Set output dir
        export_directory = "../../models/"
        full_export_directory = os.path.join(export_directory, self.model_name)

        clear_session()
        print(params)

        self.set_model(
            n_units=params["n_units"],
            add_recurrent_layer=params["add_recurrent_layer"],
            dropout=params["dropout"],
            spatial_dropout=params["spatial_dropout"],
            hidden_dense_units=params["hidden_dense_units"],
            learning_rate=params["learning_rate"],
            bidirectional=params["bidirectional"],
        )
        self.fit_model(batch_size=params["batch_size"], epochs=params["epochs"])

        loss = 1 - self.early_stopping.best

        # Keep log of best loss and save the corresponding model
        metric_file_name = os.path.join(full_export_directory, "metric.txt")
        try:
            with open(metric_file_name) as f:
                min_loss = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
            min_loss = 1000  # else just use current value as the best

        if loss < min_loss:
            print(f"Found new best model with loss {loss}... Saving model.")
            self.save_model(os.path.join(full_export_directory,"model.h5")) # save best to disc and overwrite metric
            with open(metric_file_name, "w") as f:
                f.write(str(loss))
        sys.stdout.flush()
        return {
            "loss": loss,
            "status": STATUS_OK,
            "model_history": self.history.history,
        }

    def search_hyperparameters(self, space, version, total_trials):
        shp = safeHyperopt(
            model=self.hyperopt_model,
            space=space,
            version=version,
            model_name = self.model_name,
            total_trials=total_trials,
        )

        shp.run_train_loop()

class LstmCnn(Trainer):

    def __init__(self,
                 train_data,
                 val_data,
                 embedding_dir,
                 model = None):

        self.train_data = train_data
        self.val_data = val_data
        self.embedding_dir = embedding_dir
        self.gensim_model = read_glove_file(self.embedding_dir)
        self.word_to_index,self.index_to_words = get_word_index_dicts(self.gensim_model)
        
        self.m_X, self.n_X = self.train_data['X_indices'].shape
        self.m_X_aux, self.n_X_aux = self.train_data['X_aux'].shape
        
        self.model = model
        self.model_name = type(self).__name__
        
    def set_model(self,
                         n_units = 2**7,
                         filter_size = 2,
                         n_filters = 2**6,
                         padding = 'valid',
                         stride = 1,
                         dropout = 0.1,
                         spatial_dropout = 0,
                         hidden_dense_units = 0,
                         learning_rate = 0.01,
                         bidirectional = True,
                         global_max_pool = False,
                         global_avg_pool = False,
                         seed = 100):

        #Input layer
        sentence_indices = Input((self.n_X),dtype='int32')

        # Embedding Layer
        embedding_layer = pretrained_embedding_layer(self.gensim_model,
                                                     self.word_to_index)

        # Propagate sentence_indices through your embedding layer
        embeddings = embedding_layer(sentence_indices)   
        embeddings = SpatialDropout1D(spatial_dropout,seed= seed)(embeddings)

        #LSTM 1
        if bidirectional:
            X = Bidirectional(LSTM(n_units,return_sequences=True))(embeddings)
        else:
            X = LSTM(n_units,return_sequences=True)(embeddings)

        items_to_concat = []

        #1D Convolution

        X = Conv1D(filters = n_filters, kernel_size = filter_size,strides = stride,padding = padding, activation = 'relu')(X)
        #TODO: ADD BATCH NORM ??
        if global_avg_pool:
            avg_pool = GlobalAveragePooling1D()(X)
            items_to_concat.append(avg_pool)

        if global_max_pool:
            max_pool = GlobalMaxPooling1D()(X)
            items_to_concat.append(max_pool)

        X = concatenate(items_to_concat)

        if hidden_dense_units>0:
            #Hiden Dense Layer 1
            X = Dense(hidden_dense_units)(X)
            X = BatchNormalization()(X) 
            X = Activation(activation='relu')(X)
            X = Dropout(dropout,seed = seed)(X)

        #Output layer
        X = Dense(1,activation='sigmoid')(X)

        self.model = Model(sentence_indices,X)

        opt = Adam(learning_rate=learning_rate)
        self.model.compile(loss='binary_crossentropy',metrics = [f1_metric],optimizer=opt)
    

    def save_model(self,file_path):
        self.model.save(file_path)
    

    def fit_model(self,epochs,batch_size,use_early_stopping = True,monitor = 'val_f1_metric',patience = 15, min_delta = 1e-4):

        self.early_stopping = EarlyStopping(monitor=monitor,
                                                         min_delta=min_delta,
                                                         patience = patience,
                                                         mode = 'max',
                                                         restore_best_weights = True,
                                                         verbose = 0)

        self.history = self.model.fit(self.train_data['X_indices'],
                                       self.train_data['y'],
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       validation_data = [self.val_data['X_indices'],self.val_data['y']],
                                       verbose = 0,
                                       callbacks=[self.early_stopping] if use_early_stopping else None)

    def generate_metrics(self,
                         X_test,
                         y_test,
                         metric = 'f1_score'):

        y_test = y_test.astype('float32')
        y_pred = self.model.predict(X_test)
        
        if metric == 'f1_score':
            score = f1_metric(y_test,y_pred).numpy()
        
        return score

    def hyperopt_model(self, params: dict, verbose: int = 0):
        #Set output dir
        export_directory = '../../models/'
        full_export_directory = os.path.join(export_directory,self.model_name)
        
        clear_session()
        print(params)
        
        self.set_model( n_units=params['n_units'],
                        filter_size = params['filter_size'],
                        n_filters = params['n_filters'],
                        padding = params['padding'],
                        stride = params['strides'],
                        dropout=params['dropout'],
                        spatial_dropout = params['spatial_dropout'],
                        hidden_dense_units=params['hidden_dense_units'],
                        learning_rate = params['learning_rate'],
                        bidirectional = params['bidirectional']
                        )
        self.fit_model(batch_size=params['batch_size'],
                      epochs=params['epochs'])                
                                    
        loss = 1 - self.early_stopping.best
        
        #Keep log of best loss and save the corresponding model
        metric_file_name = os.path.join(full_export_directory,'metric.txt')
        try:
            with open(metric_file_name) as f:
                min_loss = float(f.read().strip())  # read best metric,
        except FileNotFoundError:
                min_loss = 1000  # else just use current value as the best

        if loss < min_loss:
            print(f"Found new best model with loss {loss}... Saving model.")
            self.save_model(os.path.join(full_export_directory,"model.h5"))  # save best to disc and overwrite metric
            with open(metric_file_name, "w") as f:
                f.write(str(loss))
        sys.stdout.flush() 
        return {'loss': loss, 'status': STATUS_OK, 'model_history':self.history.history}

    def search_hyperparameters(self,space,version,total_trials):
        shp = safeHyperopt( model = self.hyperopt_model,
                            space = space,
                            version = version,
                            model_name = self.model_name,
                            total_trials = total_trials)

        shp.run_train_loop()
