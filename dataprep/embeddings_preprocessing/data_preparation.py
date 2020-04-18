import numpy as np
from tensorflow.keras.layers import Embedding

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def sentences_to_indices(X, word_to_index: np.array, max_len: int)->np.array:
    """
    Converts an array of tokenized and cleaned sentences into an array of indices corresponding 
    to words in the sentences. The output shape should be such that it can be given to `Embedding()`. 
    
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i]
                
        # Loop over the words of sentence_words
        for j,w in enumerate(sentence_words):
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            #If word is oov, then set to unkown
            assert j<max_len,'Max_len parameter is too small. Please increase max_len'
            try:
                X_indices[i, j] = word_to_index[w]
            except KeyError:
                X_indices[i, j] = word_to_index['<unknown>']
    return X_indices


def pretrained_embedding_layer(gensim_model, word_to_index: dict,trainable: bool = False, mask_zero:bool =True):
    """
    Creates a Keras Embedding() layer and loads in pre-trained Embedding
   
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = gensim_model.vector_size      # define dimensionality of your word vectors

    # Initialize the embedding matrix as a numpy array of zeros.
    emb_matrix = np.zeros((vocab_len,emb_dim))
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary.
    #Basically we map the index to its corresponding vector
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = gensim_model[word]

    # Define Keras embedding layer. We 
    embedding_layer = Embedding(vocab_len,emb_dim,trainable = trainable, mask_zero=mask_zero)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer