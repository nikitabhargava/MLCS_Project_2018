#Change file permissios to run the script
chmod 755 RunAllmodels.sh

# Run random forest model
python3 /data/WorkData/firmEmbeddings/Models/StockPredictionUsingRandomForest/RunRandomForest.py

#Run Neural Network model for stock prediction
python3 /data/WorkData/firmEmbeddings/Models/StockPredictionUsingNeuralNetwork/NeuralNetworkRun_3layers.py

#Run the Neura Network for firm embeddings
python3 /data/WorkData/firmEmbeddings/Models/FirmEmbeddingsModel.py
