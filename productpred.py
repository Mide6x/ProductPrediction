import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)

#load data
def load_categories(filename: str) -> dict:
    """
    Loads categories and subcategories from a JSON file and converts them into a dictionary.

    Parameters:
        - filename (str): The path to the JSON file containing categories and subcategories.

    Returns:
        - dict: A dictionary with main categories as keys and corresponding subcategories as values.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            categories_dict = {item["categories"]: item["subcategories"] for item in data}
            return categories_dict
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {filename} contains invalid JSON.")
        return {}

categories_data = load_categories("data.json")

## For Categories:
def predict_main_category(product_name: str, client: OpenAI, data: dict) -> str:
    """
    Predicts the main category of a product using the OpenAI GPT API.

    Parameters:
        - product_name (str): The name of the product.
        - client (OpenAI): An instantiated OpenAI client.
        - data (dict): A dictionary containing main categories and their subcategories.

    Returns:
        - str: Predicted main category.
    """
    # Extract main categories from the data
    main_categories = list(data.keys())

    prompt = f"""
    You are a product categorization assistant. Based on the following main categories, 
    assign the correct main category to the product.

    Main Categories:
    {', '.join(main_categories)}

    Product: "{product_name}"
    
    Provide the result in the following format:
    MainCategory: [Category]
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.5,
    )

    result = response.choices[0].message.content.strip()
    category = result.split(":")[1].strip() if "MainCategory:" in result else "Unknown"

    return category

## For Subcategories:
def predict_subcategory(product_name: str, main_category: str, client: OpenAI, data: dict) -> str:
    """
    Predicts the subcategory of a product based on a selected main category using the OpenAI GPT API.

    Parameters:
        - product_name (str): The name of the product.
        - main_category (str): The selected main category.
        - client (OpenAI): The instantiated OpenAI client.
        - data (dict): A dictionary containing categories and subcategories.

    Returns:
        - str: Predicted subcategory.
    """
    available_subcategories = data.get(main_category, [])

    prompt = f"""
    You are a product categorization assistant. Based on the main category "{main_category}" and its subcategories, 
    assign the correct subcategory to the product.

    Subcategories:
    {', '.join(available_subcategories)}

    Product: "{product_name}"

    Provide the result in the following format:
    Subcategory: [Subcategory]
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.5,
    )


    result = response.choices[0].message.content.strip()
    subcategory = result.split(":")[1].strip() if "Subcategory:" in result else "Unknown"
    
    return subcategory

## For Manufacturers
def predict_manufacturers(product_name: str, client: OpenAI) -> list:
    """
    Predicts the top 4 most probable manufacturers of a product using the OpenAI GPT API.

    Parameters:
        - product_name (str): The name of the product.
        - client (OpenAI): An instantiated OpenAI client.

    Returns:
        - list: List of the top 4 predicted manufacturers.
    """

    prompt = f"""
    You are a product categorization assistant. Based on the product name, 
    predict the top 4 most probable manufacturers for the product.

    Product: "{product_name}"

    Provide the result in the following format:
    Manufacturers: [Manufacturer1, Manufacturer2, Manufacturer3, Manufacturer4]
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.5,
    )

    result = response.choices[0].message.content.strip()
    if "Manufacturers:" in result:
        manufacturers = result.split(":")[1].strip().split(",")
        manufacturers = [m.strip() for m in manufacturers if m.strip()]
    else:
        manufacturers = []

    return manufacturers[:4]


## Predict:
if __name__ == "__main__":

    product_name = input("Enter the product name: ")

    #Category Predicion
    predicted_category = predict_main_category(product_name, client, categories_data)


    # Subcategory Prediction
    main_category = predicted_category
    predicted_subcategory = predict_subcategory(product_name, main_category, client, categories_data)

    # Manufacturer Prediction
    predicted_manufacturers = predict_manufacturers(product_name, client)

    # Show Predictions
    print(f"Predicted Main Category: {predicted_category}")
    print(f"Predicted Subcategory: {predicted_subcategory}")
    print(f"Predicted Manufacturers: {predicted_manufacturers}")