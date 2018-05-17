
We have run our scripts on the GPU server 13.90.81.78. It has all the tools and packages required by the script. 


#Model Run

Path where the models are located

Main Directory is :
/data/WorkData/firmEmbeddings/Models/

1. Subdirectory for Random Forest Algorithm is :
StockPredictionUsingRandomForest/

Running the script RunRandomForest.py will generate the model. The model will generate the plot actual/predicted at the end of the script. 

2. Subdirectory for Neural Network for Stock Prediction:
StockPredictionUsingNeuralNetwork/

Running the script NeuralNetworkRun_3layers.py will train the model. The predictions on test data are saved in predictions.txt in the same path. This file along with actual.txt will be used in plotting the actual/predicted stock price change. 

3. Sudirectory for Neural Network for firm Embeddings
FirmEmbeddings/

Running the script NeuralNetworkRun_3layers.py will generate the firm embeddings. This embedding will be saved in the same path.

