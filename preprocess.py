import re
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords

def preprocess_text(text_data):
    preprocessed_text = []

    for sentence in tqdm(text_data):
        sentence = re.sub(r"[^\w\s]", "", sentence)
        preprocessed_text.append(
            str(
                " ".join(
                    token.lower()
                    for token in str(sentence).split()
                    if token not in stopwords.words("english")
                )
            )
        )

    return preprocessed_text


def preprocess_data(file):
    data = pd.read_csv(file, index_col=0)  # read auth-news file
    data = data.drop(["title", "subject", "date"], axis=1)  # drop some non-use column
    # data.isnull().sum()  # check if there is any null value (if yes drop it)

    # shuffle the dataset to prevent the model to get bias
    data = data.sample(frac=1)
    data.reset_index(inplace=True)
    data.drop(["index"], axis=1, inplace=True)


    preprocessed_review = preprocess_text(data["text"].iloc[0:20000].values)
    preprocessed_news_list = list(zip(preprocessed_review, data["class"].iloc[0:20000]))

    df = pd.DataFrame(preprocessed_news_list, columns=["text", "class"])
    df.to_csv("Data/sample.csv", index=True)
