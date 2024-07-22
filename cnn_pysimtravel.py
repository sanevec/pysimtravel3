from io import StringIO
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, Input
from keras.optimizers import Adam
from keras.regularizers import l2

import pandas as pd
import json 

from keras import callbacks

import os
from dataSimtravel import DataCNN, DataCNN2

import numpy as np
import copy as cp

import seaborn as sns
import matplotlib.pyplot as plt

import datetime
import time

class CNNPysimtravel():

    # TODO:
    #   - Inference
    #   - Results
    #   - Logger
    #   - Probar que el splitData funciona correctamente y se hace lo que queremos.

    def __init__(self, dataShape: tuple[int, int, int, int], dataDir: str, csvDir:str, 
                 saveGraphics:bool=False, saveOutput:str = "dataSimtravelCNN", dirModels:str = "/home/josmorfig1/sanevec/source/replicate_antonio/cnnResults/models",
                 epoch:int = 50, learningRate: float = 0.001, batch_size:int = 64, alpha:float = 0.01, 
                 weight_decay:float = 0.0001, l2_value: float = 0.001, 
                 dropout_value_layer: float= 0.2, loss_function:str = "mean_squared_error", isNormalize:bool = False,
                 model_name:str = "best_model", lr_factor:float=0.5, patience:int=15) -> None:
        self.__model = Sequential()
        # self.dataShape = dataShape
        self.isTrained = False

        if not os.path.exists(saveOutput):
            print("Creando carpeta de salida en: ", saveOutput)
            os.makedirs(saveOutput)
        
        self.dataSimtravel = DataCNN2(dataDir, csvDir, saveOutput, batch_size=batch_size, isNormalize=isNormalize)
        self.saveGraphics = saveGraphics
        self.saveOutput = saveOutput


        self.epoch = epoch
        self.batch_size = batch_size
        self.learningRate = learningRate
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.l2_value = l2_value
        self.dropout_value_layer = dropout_value_layer
        self.loss_function = loss_function

        self.lr_factor = lr_factor
        self.patience = patience

        
        self.dirModels = dirModels
        i = 1
        _modelName, modelName = model_name,model_name
        while os.path.exists(os.path.join(dirModels, modelName+'.keras')):
            modelName = _modelName+'_'+str(i)
            i+=1

        _modelName = modelName
        modelName = os.path.join(dirModels, modelName+'.keras')
        self.modelName = modelName
        self._modelName = _modelName

        self.isNormalize = isNormalize

    @property
    def model(self):
        return self.__model

    @property
    def isTrainedF(self) -> None:
        if not self.isTrained:
            raise "Errorl, the mode is not trained"
        
    def showModel(self) -> Sequential:
        return self.model.summary()

    def makeModel(self) -> None:

        model = self.__model

        inputShpae = self.dataShape

        dtype = 'float32'
        alpha = self.alpha
        l2_value = self.l2_value
        weight_decay = self.weight_decay
        learningRate = self.learningRate
        loss_function = self.loss_function
        dropout_value_layer = self.dropout_value_layer

        # Init layer
        model.add(Input(shape=inputShpae))

        # First convolutional layer fase X1 (3,3)
        nLayers = 5 # use later 2**nLayers to calculate the number of filters.
        for _ in range(2):
            model.add(Conv2D(2**nLayers, (3,3), padding='same', dtype=dtype)) # l1_output_shape = (dataS[0], dataS[1], 32)
            model.add(LeakyReLU(alpha=alpha))
            model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(2,2), dtype=dtype)) # reduce dimensionality
        model.add(Dropout(dropout_value_layer))
        nLayers+=1
        
        # Second convolutional layers fase X3 (3,3)
        for _ in range(2):
            model.add(Conv2D(2**nLayers, (3,3), padding='same', dtype=dtype)) # l1_output_shape = (dataS[0], dataS[1], 32)
            model.add(LeakyReLU(alpha=alpha))
            model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(2,2), dtype=dtype)) # reduce dimensionality
        model.add(Dropout(dropout_value_layer))
        nLayers+=1

        # Third convolutional layers fase x2 (3,3)
        for _ in range(2):
            model.add(Conv2D(2**nLayers, (3,3), padding='same', dtype=dtype)) # l1_output_shape = (dataS[0], dataS[1], 32)
            model.add(LeakyReLU(alpha=alpha))
            model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(2,2), dtype=dtype)) # reduce dimensionality
        model.add(Dropout(dropout_value_layer))
        nLayers+=1

        # Fourd convolutional layers fase x2 (5,5)
        for _ in range(2):
            model.add(Conv2D(2**nLayers, (5,5), padding='same', dtype=dtype)) # l1_output_shape = (dataS[0], dataS[1], 32)
            model.add(LeakyReLU(alpha=alpha))
            model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(2,2), dtype=dtype)) # reduce dimensionality
        model.add(Dropout(dropout_value_layer))

        for _ in range(2):
            model.add(Conv2D(2**nLayers, (5,5), padding='same', dtype=dtype)) # l1_output_shape = (dataS[0], dataS[1], 32)
            model.add(LeakyReLU(alpha=alpha))
            model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(2,2), dtype=dtype)) # reduce dimensionality
        model.add(Dropout(dropout_value_layer))
        nLayers+=1
        
        for _ in range(1):
            model.add(Conv2D(2**nLayers, (5,5), padding='same', dtype=dtype)) # l1_output_shape = (dataS[0], dataS[1], 32)
            model.add(LeakyReLU(alpha=alpha))
            model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(2,2), dtype=dtype)) # reduce dimensionality
        model.add(Dropout(dropout_value_layer))
        nLayers+=1
        
        # Flat the l4_output
        model.add(Flatten(dtype=dtype))

        # Dense layers fase x3
        _neurons = 2**nLayers
        for i in range(4):
            reducedValue = 2**i
            neurons = _neurons/reducedValue
            model.add(Dense(neurons, dtype=dtype))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(.5, dtype=dtype))
        # output layer
        model.add(Dense(1, activation='linear', dtype=dtype)) # linear to get the regresion

        optimizer = Adam(learning_rate=learningRate, decay=weight_decay)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['mean_absolute_error'])

        self.__model = model
        return self.model
    
    def getStrSummaryModel(self):
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n\t\t'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string

    def saveJsonModel(self, fileToSave="C:\\Users\josem\\Jose\\Sanevec\\pysimtravel3\\models\\configurations.json"):
        hyp_params = {
                    "epoch":self.epoch,
                    "batch_size":self.batch_size,
                    "learningRate": self.learningRate,
                    "alpha" : self.alpha,
                    "weight_decay" : self.weight_decay,
                    "l2_value" : self.l2_value,
                    "dropout_value_layer" : self.dropout_value_layer,
                    "loss_function" : self.loss_function,
                    "lr_factor" : self.lr_factor,
                    "patience" : self.patience
                }
        data = {
            self._modelName : {
                "hyp_params" : hyp_params,
                "summaryModel":self.getStrSummaryModel()
            }
        }
        with open(fileToSave, 'a') as j:
            json.dumb(data, j, ident=4)

    def saveTextModel(self, fileToSave="C:\\Users\josem\\Jose\\Sanevec\\pysimtravel3\\models\\configurations.txt"):
        with open(fileToSave, 'a') as f:
            results = f"\n\t\ttest_loss={self.test_loss}, test_mae={self.test_mae}"
            hypParams = f"\n\t\tepoch={self.epoch}, batch_size={self.batch_size}, learningRate={self.learningRate}, alpha={self.alpha}, l2_value={self.l2_value}, weight_decay={self.weight_decay}, dropout_value_layer={self.dropout_value_layer}, loss_function={self.loss_function}, isNormalize={self.isNormalize}, patience={self.patience}, lr_factor={self.lr_factor}\n"
            w = f"{datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y %H:%M:%S')}- {self._modelName}: \n\thypeParams={hypParams}\n\tresults={results}\n\t{self.getStrSummaryModel()}{'#'*50}\n"
            f.write(w)

    def splitData(self) -> None:
        print("Preparando los datos...")
        self.dataSimtravel.prepareData()
        train_dataset, val_dataset, test_dataset = self.dataSimtravel.getDatasets()
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        print("Shape: ")
        print(self.train_dataset.element_spec)
        self.dataShape = self.dataSimtravel.dataShape
        print(self.dataShape)
        print("Datos finalizados")
    
    def train(self) -> None:
        model = self.__model
        # Only save the best model

        checkpoint_cb = callbacks.ModelCheckpoint(self.modelName, save_best_only=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=self.lr_factor, patience=5, min_lr=0.0001)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=False)


        train_data = self.train_dataset
        val_dataset = self.val_dataset

        history = model.fit(
            train_data,
            epochs = self.epoch,
            batch_size = self.batch_size,
            validation_data = val_dataset,
            callbacks = checkpoint_cb,
            verbose=2
        )

        self.__model = model
        self.history=history
        print("Finish train")
        self.isTrained = True
        return history

    def evaluate(self) -> None:
        """Evaluate the model with the test data.

            raise -> If the data hasn't been splited correctly.
        """

        if not hasattr(self, "test_dataset"):
            raise "Error, no se ha preparado el dataset previamente."
        
        test_data = self.test_dataset
        test_loss, test_mae = self.__model.evaluate(test_data)
        self.test_loss, self.test_mae = test_loss, test_mae

        print(f'Mean Absolute Error on test set: {test_mae}')
        print(f'Loss on test set: {test_loss}')
        return test_loss, test_mae

    def inference(self, _input, model="best_model.keras") -> None:
        self.isTrainedF
        if isinstance(_input, list):
            pass
        else:

            pass
        pass
    
    def copy(self):
        return cp.deepcopy(self)
    
    def showGraphicTrain(self):
        history = self.history
        sns.set(style="whitegrid")

        # Obtener datos del historial
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mean_absolute_error']
        val_mae = history.history['val_mean_absolute_error']
        epochs = range(1, len(loss) + 1)

        # Crear un DataFrame para facilitar el plotting con Seaborn
        
        data = {
            'Epoch': epochs,
            'Training Loss': loss,
            'Validation Loss': val_loss,
            'Training MAE': mae,
            'Validation MAE': val_mae
        }
        df = pd.DataFrame(data)

        # Graficar pérdida
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.lineplot(x='Epoch', y='Training Loss', data=df, marker='o', label='Training Loss')
        sns.lineplot(x='Epoch', y='Validation Loss', data=df, marker='o', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Graficar métricas (MAE)
        plt.subplot(1, 2, 2)
        sns.lineplot(x='Epoch', y='Training MAE', data=df, marker='o', label='Training MAE')
        sns.lineplot(x='Epoch', y='Validation MAE', data=df, marker='o', label='Validation MAE')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        self.saveImage(f"{self._modelName}_graphicTrain", self.dirModels, plt)
        plt.show()

    def showGraphicTest(self, modelPath = None):

        if modelPath is None:
            modelPath = self.modelName # use the best model trained

        model = tf.keras.models.load_model(modelPath)
        yPredict = abs(model.predict(self.test_dataset).flatten())
        yReal = []
        for batch in self.test_dataset:
            yReal.extend(abs(batch[1].numpy()))
        yReal = np.array(yReal).flatten()

        print(yPredict, " - ", yReal)
        
        plt.figure(figsize=(10,10))
        sns.scatterplot(x=yPredict, y=yReal)

        min_val= min([*yPredict, *yReal, 0])
        max_val= max([*yPredict, *yReal])

        plt.plot([min_val, max_val], [min_val, max_val], label="Reacta ideal")

        plt.axis('equal')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.xlabel("Valor predicho")
        plt.ylabel("Valor real")
        plt.title("Comparación real vs predicción")

        plt.legend()
        self.saveImage(f"{self._modelName}_graphicTest", self.dirModels, plt)
        plt.show()
    
    def normalizeLabels(self, xValues, yValues):
        return xValues, (yValues-self.y_train_mean)/self.y_train_std

    @staticmethod
    def saveImage(name, outputdir, image):
        outputdir = os.path.join(outputdir, 'figures')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        filename=os.path.join(outputdir, name)
        image.savefig(filename)

def penaltyLowVar(y_true, y_pred, lambda_var=1.0):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    variance = tf.math.reduce_variance(y_pred)
    var_penalty = lambda_var / (variance + 1e-6)  # Añadir un pequeño valor para evitar división por cero
    loss = mse + var_penalty
    return loss


    