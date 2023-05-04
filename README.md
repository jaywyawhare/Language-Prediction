# Multilingual Text Classification using Deep Learning Techniques
This project presents a comparative study of language detection and classification approaches using deep learning techniques and embedding visualization. The study utilized a dataset consisting of text and language columns with 17 different languages. In-built modules such as LangDetect, LangId, and FastText were used for language detection, and Sentence Transformer was used for embedding. The embeddings were then visualized using t-SNE to reduce dimensionality, and multi-layer perceptron model, LSTM, and Convolution were used for classification. Two types of embeddings were compared: FastText embeddings with a dimensionality of 16 and Sentence Transformer embeddings with a dimensionality of 384.

## Dataset
The dataset used in this project contains text and language columns with 17 different languages. The text column contains a random selection of sentences in each language, while the language column specifies the language of each sentence. The dataset is available in the data folder of this repository.

## Requirements
The project requires the following packages to be installed:

- Pandas
- NumPy
- Scikit-learn
- LangDetect
- LangId
- FastText
- Sentence Transformer
- TensorFlow
- Matplotlib
- Seaborn

To install these packages, you can run the following command:

```bash
> pip install pandas numpy scikit-learn langdetect langid fasttext sentence-transformers tensorflow matplotlib seaborn
```

To run the project, you can follow these steps:

- Clone the repository to your local machine:
```bash
> git clone https://github.com/jaywyawhare/Language-Prediction.git
```

- Navigate to the project directory:
```bash
> cd Language-Prediction
```

- Install the required packages:

```bash
> pip install -r requirements.txt
```

- To deploy it:
```bash
> streamlit run main.py
```

The script will load the dataset, preprocess the data, train and evaluate the models, and display the results.

## Results
The results of the study show that the FastText multi-layer perceptron model achieved the highest accuracy, precision, recall, and F1 score of 0.9985, 0.9962, 0.9961, and 0.9961, respectively. On the other hand, the Sentence Transformer multi-layer perceptron model achieved an accuracy of 0.9342, precision of 0.9365, recall of 0.9342, and F1 score of 0.9324. The results indicate that the dimensionality of the embeddings played a significant role in the clustering of languages, with FastText embeddings showing clear clustering in the 2D visualization due to its training on a large multilingual corpus.

## Conclusion
The study demonstrates the effectiveness of deep learning techniques and embedding visualization in multilingual text classification. The results provide insights into the importance of using a large multilingual corpus for training the embeddings and highlight the performance differences between the two types of embeddings. The project can be useful for practitioners interested in developing language detection and classification systems, and it provides a foundation for future research in the field of multilingual text classification.