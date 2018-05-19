
We have run our scripts on the GPU server 13.90.81.78.

# To install the packages for running all the scripts execute the command-
pip3 install -r requirements.txt

# Raw data for cases and stock price change can be found at path-
    # Main Directory 
        /data/WorkData/firmEmbeddings/
        
    # Case Data is inside the directory
        CaseData/
        
    # Stock Data is inside the directory
        StockData/
        
# The data after processing and joining can be found at path - 
    # Main Directory 
        /data/WorkData/firmEmbeddings/Models/
        
    # Random Forest for Stock Prediction Data
         StockPredictionUsingRandomForest/
         
    # Neural Network for Stock Prediction Data
        StockPredictionUsingNeuralNetwork/
        
    # Neural Network for Firm Embeddings Data
        FirmEmbeddings/
        
        
# Script to process the raw case data is -


# Script to process the raw Stock Data is - 


# Script to join the two data sets - 

 
# Script to generate models  for stock prediction and firm embeddings -

    #Change file permissios to run the script
    chmod 755 RunAllmodels.sh
    
    # Run the following command to execute the script -
    sh RunAllmodels.sh

