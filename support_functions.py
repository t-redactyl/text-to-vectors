# Contains a series of auxillary functions for text processing notebooks
import pandas as pd
import numpy as np
import spacy
import en_core_web_sm

from keras import models, layers


def apply_string_cleaning(dataset: pd.Series) -> pd.Series:
    """
    Applies a series of string cleaning tasks to a Pandas Series containing string data. The following cleaning
    steps are applied:
    - Convert all text to lowercase
    - Remove strings starting with @ (tags), # (hashtags), `r/` (Reddit sub reference)
      or `u/` (Reddit user reference).
    - Remove all non-alphabetic characters
    - Remove all single character words
    - Remove all whitespace
    """

    return (
        dataset
        .str.lower()
        .str.replace("@\w+", "", regex=True)
        .str.replace("#\w+", "", regex=True)
        .str.replace("\s[u|r]/\w+", "", regex=True)
        .str.replace("[^a-zA-Z]", " ", regex=True)
        .str.replace(r"\b\w\b", "", regex=True)
        .str.replace("\s+", " ", regex=True)
        .str.strip()
    )


def train_text_classification_model(
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
        input_size: int,
        num_epochs: int,
        hidden_layer_size: int) -> models.Sequential:
    model = models.Sequential()
    model.add(layers.Dense(hidden_layer_size, activation="relu", input_shape=(input_size,)))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"]
                  )

    model.fit(train_features,
              train_labels,
              epochs=num_epochs,
              batch_size=1920,
              validation_data=(validation_features, validation_labels)
              )
    return model


def generate_predictions(model: models.Sequential,
                         validation_features: np.ndarray,
                         validation_labels: np.ndarray) -> list:
    predicted_proba = model.predict(validation_features)
    predicted_labels = [sl for l in np.rint(predicted_proba) for sl in l]

    print(pd.crosstab(validation_labels, predicted_labels))
    return predicted_labels


nlp = spacy.load("en_core_web_sm")


def lemmatise_text(texts: pd.Series) -> pd.Series:
    lemmatised_texts = []
    for doc in nlp.pipe(texts):
        lemmatised_texts.append(" ".join([token.lemma_ for token in doc]))
    return pd.Series(lemmatised_texts)
