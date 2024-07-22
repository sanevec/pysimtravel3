from cnn_pysimtravel import *
import itertools

if __name__ == "__main__":
    
    dataDir="/home/josmorfig1/sanevec/source/replicate_antonio/datas/dataCNN_20000_formated"
    csvDir=dataDir+"/results_cnnExp.csv"
    saveOutput=f"/home/josmorfig1/sanevec/source/replicate_antonio/cnnResults/models/"

    print("Iniciando el test de la CNN para la predicción del fit de los modelos.")

    dataShape=(288,288,4)
    batch_size=256
    epoch=80
    learningRate = 0.001
    alpha = 0.01
    l2_value = 0.001
    dropout_value_layer = 0.3
    weight_decay = 0.0001
    loss_function = "mean_squared_error"
    modelName = "testArchitecture"
    lr_factor = 0.5
    patience = 10
    lambda_var = 1.

    isNormalize = True
    useGridSearch = True
    
    def f_loss(y_true, y_pred):
        return penaltyLowVar(y_true, y_pred, lambda_var)

    if useGridSearch:
        epoch=[80] # max times to train the model with trainDS
        batch_size=[256, 512] # Number of example per steps each epoch
        learningRate=[1e-3, 5e-3, 1e-4] # How much the model learn in each step
        alpha=[1e-2]
        l2_value=[1e-3] 
        weight_decay=[1e-4] 
        dropout_value_layer=[2e-1,3e-1] # How decrease the lr for each 
        loss_function=["mean_squared_error", "mean_absolute_error"]
        isNormalize=[False]
        lr_factor=[5e-1]
        modelName = "testHypParams"

        
        hyperParams = dict(
            epoch=epoch, 
            batch_size=batch_size, 
            learningRate=learningRate, 
            alpha=alpha, 
            l2_value=l2_value, 
            weight_decay=weight_decay, 
            dropout_value_layer=dropout_value_layer,
            loss_function=loss_function, 
            isNormalize=isNormalize, 
            lr_factor=lr_factor
            )

        combinatoria = ([len(v) for v in hyperParams.values()])
        r = 1
        for c in combinatoria:
            r*=c
        print("Total de combinaciones a probar: ", r)

        keys, values = zip(*hyperParams.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for hyp_params in param_combinations:
            print("Hyp_params")
            print(hyp_params)
            model = CNNPysimtravel(dataShape=dataShape, dataDir=dataDir, csvDir=csvDir, saveOutput=saveOutput,
            model_name=modelName, patience=patience, **hyp_params)

            model.splitData()
            model.makeModel()
            model.train()
            model.evaluate()

            model.saveTextModel("/home/josmorfig1/sanevec/source/replicate_antonio/cnnResults/configurations.txt")
            model.saveJsonModel("/home/josmorfig1/sanevec/source/replicate_antonio/cnnResults/configurations.json")
            model.showGraphicTrain()
            model.showGraphicTest()

    else:
        model = CNNPysimtravel(dataShape=dataShape, dataDir=dataDir, csvDir=csvDir, 
                            saveOutput=saveOutput, epoch=epoch, batch_size=batch_size, 
                            learningRate=learningRate, alpha=alpha, l2_value=l2_value, 
                            weight_decay=weight_decay, dropout_value_layer=dropout_value_layer,
                            loss_function=loss_function, isNormalize=isNormalize, model_name=modelName,
                            patience=patience, lr_factor=lr_factor)

        model.splitData()
        model.makeModel()
        model.train()
        model.evaluate()

        model.saveTextModel("/home/josmorfig1/sanevec/source/replicate_antonio/cnnResults/configurations.txt")
        model.saveJsonModel("/home/josmorfig1/sanevec/source/replicate_antonio/cnnResults/configurations.json")
        model.showGraphicTrain()
        model.showGraphicTest()
    