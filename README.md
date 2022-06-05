# Music_Classification
Project for machine learning course


Step 0 : Download the code 
 git clone https://github.com/mbolleng2303/Music_Classification.git

Step 1 : install requirements 

pip install -r requirements.txt

Step 2 :
- Download the three dataset needed : Subset of the MSD, tagtraum genre annotations dataset,  last.fm dataset at "http://millionsongdataset.com/"
- download the folder "pythonSrc" in "https://github.com/tbertinmahieux/MSongsDB" and place it in the repository
- Open the notebook "Data_Preprocess"
- Change all the directory in the third cell : 
    - Subset MSD dataset : msd_subset_path
    - Lastfm dataset : msd_label_data
    - path : Path to the script python called : "hdf5_getters.py"
    - database_file : folder 
When you finish the run, you should see four numpy arrays related to tagtraum, it is the data ussed in the CNN



Step 3 : 

go to the first line of the main function in main.py 
and set the config path 

Step 4 : 

go into configs.json file and set the out and data path 

Step 5 : 

go to data/generate data and generate the data
modify the dimension of receive csv file in data/Dataset.py/Music2Data

(if you want to create data, please keep the same order (Nsamples,Channel, Height, Width)


Step 6 : 
run the main and change the parmeters in the config file
