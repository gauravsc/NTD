import numpy as np
from keras.preprocessing import sequence
from keras.layers import Merge, Input, Dense, merge
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print _val_f1, _val_precision, _val_recall
        return val_f1


class CNN:

    def __init__(self, preprocessor_text, output_size):
        self.preprocessor_text = preprocessor_text
        self.ngram_filters = [3, 4, 5]
        self.nb_filter = 150 
        self.dropout = 0.2
        self.output_size = output_size
        self.build_model()
        self.metrics = Metrics()
        self.metrics.on_train_begin()

    def build_model(self):
        text_input = Input(shape=(self.preprocessor_text.maxlen,), dtype='int32')
        x = Embedding(output_dim=200, input_dim=self.preprocessor_text.max_features, 
            input_length=self.preprocessor_text.maxlen, weights=self.preprocessor_text.init_vectors)(text_input)
        x = Dropout(self.dropout)(x)
        y1 = Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=2,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=200,
                                         input_length=self.preprocessor_text.maxlen)(x)
        y1 = MaxPooling1D(pool_length=self.preprocessor_text.maxlen - 2 + 1)(y1)
        y1 = Flatten()(y1)
        y2 = Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=3,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=200,
                                         input_length=self.preprocessor_text.maxlen)(x)
        y2 = MaxPooling1D(pool_length=self.preprocessor_text.maxlen - 3 + 1)(y2)
        y2 = Flatten()(y2)
        y3 = Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=4,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=200,
                                         input_length=self.preprocessor_text.maxlen)(x)
        y3 = MaxPooling1D(pool_length=self.preprocessor_text.maxlen - 4 + 1)(y3)
        y3 = Flatten()(y3)
        x = merge([y1, y2, y3], mode='concat')
        x = Dropout(self.dropout)(x)
        x = Dense(200, input_dim=self.nb_filter * len(self.ngram_filters))(x)
        output = Dense(self.output_size, activation='sigmoid')(x)
        self.model = Model(input=[text_input], output=[output])
        self.model.compile(loss='binary_crossentropy', optimizer="rmsprop")
        print("model built")
        print(self.model.summary())

    def train(self, X_text, y_train, X_text_val=None, y_val=None,
                nb_epoch=5, batch_size=32, optimizer='adam'):

        checkpointer = ModelCheckpoint(filepath="weights.hdf5", 
                                       verbose=1, 
                                       save_best_only=(X_text_val is not None))

        if X_text_val is not None:
            self.model.fit([X_text], [y_train],
                batch_size=batch_size, nb_epoch=nb_epoch,
                validation_data=([X_text_val],  [y_val]),
                verbose=2, callbacks=[self.metrics])
            self.model.load_weights('weights.hdf5')
        else: 
            print("no validation data provided!")
            self.model.fit([X_text], [y_train],
                batch_size=batch_size, nb_epoch=nb_epoch, 
                verbose=2, callbacks=[checkpointer])
        
    def predict(self, X_text, batch_size=32, binarize=False):
        raw_preds = self.model.predict([X_text], batch_size=batch_size)
        return raw_preds

class Preprocessor:
    def __init__(self, max_features, maxlen, embedding_dims=300, wvs=None):
       
        self.max_features = max_features  
        self.tokenizer = Tokenizer(nb_words=self.max_features)
        self.maxlen = maxlen  

        self.use_pretrained_embeddings = False 
        self.init_vectors = None 
        if wvs is None:
            self.embedding_dims = embedding_dims
        else:
           
            self.use_pretrained_embeddings = True
            self.embedding_dims = len(wvs[wvs.keys()[0]])
            self.word_embeddings = wvs


    def preprocess(self, all_texts):
        self.raw_texts = all_texts
        self.fit_tokenizer()
        if self.use_pretrained_embeddings:
            self.init_word_vectors()

    def fit_tokenizer(self):
        self.raw_texts = [a.replace(",", " ") for a in self.raw_texts]
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        print "all word indices: ", len(self.tokenizer.word_index.items())
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token


    def build_sequences(self, texts):
        X = list(self.tokenizer.texts_to_sequences_generator(texts))
        X = np.array(pad_sequences(X, maxlen=self.maxlen))
        return X

    def init_word_vectors(self):
        self.init_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.max_features:
                try:
                    self.init_vectors.append(self.word_embeddings[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.embedding_dims)*-2 + 1

                    self.init_vectors.append(unknown_words_to_vecs[t])

        self.init_vectors = [np.vstack(self.init_vectors)]
