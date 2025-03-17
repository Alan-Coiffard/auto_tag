#How to use

create a dataset
- take a look at "tags" :
this will be the list of tags for your ai
you can modify it by adding/deleting/modifying categories, subcategories or tags 
- take a look in bing_search : this function will download a dataset of image using the structure of "tags.json"
- change the output dir if needed
- make sure you don't modify tags.json name else you want to modify the opened file
you can also provide with a bigger numbers of download
- Once you donne with the setup of your dataset structurre and fetching its content, you can create it. To do this you launch label.py (make sure you have the right folder to read)
- Then you use to_dataframe.py to create the .csv which will be used by you trainning
Once you have that you're ready to train your AI
To do so launch "train.py"
what it will do is : first split your dataset in two
- one for the trainning 
- one for the testing
(make sure you have a big enough dataset)
then it will iterate the trainning 10 time (default var). It will then switch to test mode and test itself on the second part of the dataset finally it will show you the f1-score which indicate the percentage of fiability
There you can choose to save the model by giving it a name


- tags.json
- bing_search.py
- label.py
- to_dataframe.py
- split_dataset.py
- train.py