import joblib

# Load the saved vectorizer and model
vectorizer = joblib.load('bag_of_words_vectorizer.pkl')
model = joblib.load('svm_model.pkl')

def predict(query):
    """
    Predicts the category of a given query using a pre-trained model and vectorizer.

    This function transforms the input query into a vector representation using a
    pre-loaded vectorizer, then uses a pre-trained model to predict its category.

    Parameters:
    query (str): The input text query to be categorized.

    Returns:
    str: The predicted category for the input query.

    Note:
    This function assumes that a vectorizer and a model have been previously
    loaded into the global variables 'vectorizer' and 'model' respectively.
    """
    # Transform the query using the loaded vectorizer
    query_vec = vectorizer.transform([query])
    
    # Predict the category using the loaded model
    prediction = model.predict(query_vec)
    
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Example query
    new_query = "how r u ?"

    # Predict the category
    result = predict(new_query)
    
    print(f"The predicted category for the query '{new_query}' is: {result}")