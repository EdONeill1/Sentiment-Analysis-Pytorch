* Dataset that was used was the B2 dataset.

* BERT was used a reference. It was ran on Google Collab thus all of the code to Google Collab such as mounting the drive is present in the notebook. The saved models for BERT can be found at the link : https://drive.google.com/drive/folders/1eySiJ_j9gdMysmyZyMg65seQdZiqcp5z?usp=sharing

The name of the folder with the BERT models is Bert_Models_17342691

The reason why I had to upload them to Google Drive was because they were too big causing an error in uploading to CSMoodle. I double checked with Guenole and Nikita and they allowed this. Please let me know if there is any issue accessing these saved models.

* Models are in their respective notebooks.

* Models weights are saved in the folder Experiments

* Paths in which Tabular datasets are saved to is a variable called PATH. I was unable to construct the program with a relative path. Please change where necessary. The path which the logs are saved to is also this variable PATH. The path in which the clean datasets are saved to is also this PATH variable which needs to be set with the full path.

* No paths are associated with the train and test csv files. Please change where appropriate or move the train and test csv files to the folder where the model notebooks are stored.

* The code to save the logs are commented out.

* To install the requirements, please open the 'Install_Requirements' notebook and run the cell in it or run the command : !pip install -r requirements.txt

* Please note that the test loss results of Bert are presented in decimal instead of percentage in the BERT notebook.

* All models of all experiments are given but the best ones are given in their own folder called Models_Best

* Please note that folder with the Best Models are the models with the best hyper parameters. These best hyper parameters for all are learning rate and weight decay of 1e-3, and an embedding dimension of 128. For the RNN the best max vocabulary size was 50,000 and hidden dimension of 64 and a batch size of 64, for the LSTM the best max vocabulary size was 25,000 and a hidden dimension of 128 and a batch size of 64. For BERT the batch size was 32 and hidden dimension was 64. 

* Please note that in each notebook, each model is in their own cell because the SEED wouldn't set if they were in different cells.

* On a high level, the programs work in the following way :
 - CSV files are read and the data is cleaned and saved into new CSV files.
 - These files are made into Tabular Datasets which are then used in the creation of batches and iterators.
 - The model class is defined.
 - The model trains and performs an evaluation at the end. 

* Sometimes when the notebook runs, a future warning appears. I don't know what effect this has on the training but if you stop the cell and rerun, it should go away.


Please contact me at 'edward.oneill@ucdconnect.ie' if there are any issues running the code such as BERT from Google Collab to a notebook or PATH issues but there should not be any problems.
