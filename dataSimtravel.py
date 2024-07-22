import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np

class DataCNN:

    def __init__(self, imageDir:str, csvFile:str, outputDir:str, trainPct:int=.7, valPct:int=.15, testPct:int=.15, batch_size=64) -> None:
        """Class to split the data an save correctly

        Args:
            imageDir (str): dir for the raw images
            csvFile (str): dir for the raw csv
            outputDir (str): main dir to save the new split
            trainPct (int, optional): percentage for the train split. Defaults to .7.
            valPct (int, optional): percentage for the validation split. Defaults to .15.
            testPct (int, optional): percentage for the test split. Defaults to .15.
        """
        self.imageDir = imageDir
        self.csvFile = csvFile
        self.outputDir = outputDir
        self.trainPct = trainPct
        self.valPct = valPct
        self.testPct = testPct
        self.batch_size=batch_size

    def copy_files(self, data:pd.DataFrame, subset:str) -> None:
        """Move the files to the correct directions

        Args:
            data (pd.DataFrame): filter data for the subset
            subset (str): type of subset (train, val, test)
        """
        for _, row in data.iterrows():
            src = os.path.join(self.imageDir, row['names']+'.npy')
            dst = os.path.join(self.outputDir, subset, row['names']+'.npy')
            shutil.copyfile(src, dst)
    
    def splitDataset(self) -> None:
        """Split and prepare the data obtained from the simulations. 

        Raises:
            ValueError: The csv data no has the desired columns
        """
        data = pd.read_csv(self.csvFile)
        if 'names' not in data.columns or 'labels' not in data.columns:
            raise ValueError("El CSV debe contener las columnas 'name' y 'labels'")

        # Dividir el dataset en entrenamiento, validación y prueba
        train_val_data, self.test_data = train_test_split(data, test_size=self.testPct)
        self.train_data, self.val_data = train_test_split(train_val_data, test_size=self.valPct / (self.trainPct + self.valPct))

        # self.train_data = np.expand_dim(self.train_data, axis=0) if self.train_data.ndim==3 else self.train_data
        # self.val_data = np.expand_dim(self.val_data, axis=0) if self.val_data.ndim==3 else self.val_data
        # self.test_data = np.expand_dim(self.test_data, axis=0) if self.test_data.ndim==3 else self.test_data
        
        # print("Shape: ", self.train_data.shape)
        # Crear directorios de salida
        os.makedirs(os.path.join(self.outputDir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.outputDir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.outputDir, 'test'), exist_ok=True)

        self.copy_files(self.train_data, 'train')
        self.copy_files(self.val_data, 'val')
        self.copy_files(self.test_data, 'test')

    def saveAndPrepareData(self) -> None:
        """
        Save the data splited and prepared with the function splitDataset().
        """
        self.splitDataset()

        self.train_data.to_csv(os.path.join(self.outputDir, 'train_labels.csv'), index=False)
        self.val_data.to_csv(os.path.join(self.outputDir, 'val_labels.csv'), index=False)
        self.test_data.to_csv(os.path.join(self.outputDir, 'test_labels.csv'), index=False)

        self.lenTrain, self.lenVal, self.lenTest = len(self.train_data), len(self.val_data), len(self.test_data)
        print(f'Dataset split completed: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test images.')


    def parse_matrix(self, filename, label):
        # Leer la matriz desde el archivo
        matrix = np.load(filename.numpy().decode('utf-8'))
        matrix = matrix.astype(np.float32)
        matrix = np.expand_dims(matrix, axis=0) if matrix.ndim==3 else matrix
        return matrix, label

    def tf_parse_matrix(self, filename, label):
        return tf.py_function(self.parse_matrix, [filename, label], [tf.float32, tf.float32])

    def load_data(self, image_dir, label_file):
        # Leer el CSV
        data = pd.read_csv(label_file)
        
        # Crear listas de rutas de archivos y etiquetas
        filenames = [os.path.join(image_dir, fname+".npy") for fname in data['names']]
        labels = data['labels'].values.astype('float32')
        
        if label_file == 'train':
            self.train_label_mean = np.mean(labels)
            self.train_label_std = np.std(labels)

        # Crear un dataset de TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self.tf_parse_matrix, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def prepareDataset(self,):

        self.saveAndPrepareData()
         # Cargar los datasets
        train_dataset = self.load_data(os.path.join(self.outputDir, 'train'), os.path.join(self.outputDir, 'train_labels.csv'))
        val_dataset = self.load_data(os.path.join(self.outputDir, 'val'), os.path.join(self.outputDir, 'val_labels.csv'))
        test_dataset = self.load_data(os.path.join(self.outputDir, 'test'), os.path.join(self.outputDir, 'test_labels.csv'))
        
        # Barajar, agrupar en lotes y prefetch
        self.train_dataset = train_dataset.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.val_dataset = val_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.test_dataset = test_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return self.train_dataset, self.val_dataset, self.test_dataset
    

class DataCNN2:

    def __init__(self, imageDir:str, csvFile:str, outputDir:str, isNormalize:bool = False, dataShape: tuple=(288, 288, 4), trainPct:int=.7, valPct:int=.15, testPct:int=.15, batch_size=64) -> None:
        self.data_df = pd.read_csv(csvFile)
        self.outputDir = outputDir
        self.imageDir = imageDir
        self.batchSize = batch_size

        train_val_df, self.test_df = train_test_split(self.data_df, test_size=testPct, random_state=42)
        self.train_df, self.val_df = train_test_split(train_val_df, test_size=valPct/(trainPct + valPct), random_state=42)

        self.dataShape = dataShape

        self.dirCsvSplitDir = {
            "train": os.path.join(self.outputDir, 'train_data.csv'),
            "val": os.path.join(self.outputDir, 'var_data.csv'),
            "test": os.path.join(self.outputDir, 'test_data.csv')
        }
        self.isNormalize = isNormalize

    def prepareData(self):
        self.saveSplitDF()

        self.train_dataset = self.create_dataset(self.dirCsvSplitDir['train'])
        self.val_dataset = self.create_dataset(self.dirCsvSplitDir['val'])
        self.test_dataset = self.create_dataset(self.dirCsvSplitDir['test'])

    def getDatasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def saveSplitDF(self):
        self.train_df.to_csv(self.dirCsvSplitDir["train"])
        self.val_df.to_csv(self.dirCsvSplitDir["val"])
        self.test_df.to_csv(self.dirCsvSplitDir["test"])


    def load_matrix_from_file(self, filename, label):
        # Cargar la matriz desde un archivo .npy
        filename = os.path.join(self.imageDir, filename.numpy().decode('utf-8')+'.npy')
        matrix = np.load(filename)
        return matrix, label

    def tf_load_matrix_from_file(self, filename, label):
        # Convertir a tensor y cargar la matriz
        matrix, label = tf.py_function(self.load_matrix_from_file, [filename, label], [tf.float32, tf.float32])
        # Especificar la forma de salida de los tensores
        matrix.set_shape(self.dataShape)  # Ajustar según tus dimensiones
        label.set_shape([])
        return matrix, label

    def normalize_labels(self, fileName, label):
        # if self.train_label_mean is not None and self.train_label_std is not None:
        label = (label - self.train_label_mean) / self.train_label_std
        return fileName, label
    
    def create_dataset(self, csv_path):
        # Leer el CSV en un dataset de TensorFlow
        dataset = tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=1,  # Procesar de uno en uno para mapear los nombres de archivo
            num_epochs=1,
            shuffle=False
        )
        # Quitar la dimensión adicional del batch
        dataset = dataset.map(lambda x: (x['names'][0], x['labels'][0]))
        
        if 'train' in csv_path:
            labels = []
            for _, label in dataset:
                labels.append(label.numpy())
            labels = np.array(labels)
            self.train_label_mean = np.mean(labels)
            self.train_label_std = np.std(labels)

        
        if self.isNormalize:
            dataset = dataset.map(self.normalize_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Mapear el nombre del archivo a la matriz cargada
        dataset = dataset.map(self.tf_load_matrix_from_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batchSize)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

