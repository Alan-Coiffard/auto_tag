# AI Tag Trainer

This project aims to create a dataset for training an AI model to tag pictures.

## How It Works
To train an AI model, you first need to create a dataset. This process involves the following steps:

1. **Fetch a list of URLs** for each tag based on the desired `SAMPLE_SIZE`.
2. **Download the images** using the collected URLs.
3. **Create the dataset** from the downloaded images.
4. **Train the AI model** using the generated dataset.

## Steps to Generate the Dataset

1. **Create a JSON file** listing all the tags (see `english_tags.json`).
2. **Configure the `.env` file** to set the required values. Modify them if necessary.
3. **Fetch URLs** by running:
   ```sh
   python fetch_urls.py
   ```
4. **Download images** by running:
   ```sh
   python download.py
   ```
5. **Verify the number of downloaded files** per tag by running:
   ```sh
   python verify_files.py
   ```
   - If the count is incorrect, check the `json/list_images.json` file and make necessary adjustments.
6. **Create the dataset**:
   - Run the labeling script:
     ```sh
     python label.py
     ```
   - Convert data into a structured format:
     ```sh
     python to_dataframe.py
     ```
   - Split the dataset into training and testing subsets:
     ```sh
     python split_dataset.py
     ```

## Output
Once all steps are completed, you will have a fully prepared dataset, including:
- Image files
- `dataset.csv`
- `train.csv` and `test.csv` datasets

## Train your model

Finally, **train the AI model** by running:
   ```sh
   python train.py
   ```

This will generate a trained model that you can use later.

You are now ready to train a new AI model!


## Authors

- [@Alan Coiffard](https://www.github.com/Alan-Coiffard)

