Retail Price Optimization Predictor
This project provides a basic web application for predicting retail unit prices using a machine learning model. It consists of a Flask backend that serves the HTML frontend and handles prediction requests, and an HTML/JavaScript frontend for user interaction.

Project Files
app.py: The Flask backend application.

index.html: The web page frontend (located in the templates/ directory).

random_forest_regressor_model.pkl: The trained machine learning model (RandomForestRegressor).

retail_price.csv: The dataset used for training the model and for dynamically populating product categories in the frontend.

How to Run the Application
Follow these steps to set up and run the Retail Price Optimization Predictor on your local machine:

1. Save the Files
Ensure you have the following file structure:

your-project-folder/
├── app.py
├── random_forest_regressor_model.pkl
├── retail_price.csv
└── templates/
    └── index.html

Save the Python code as app.py.

Save the HTML code (provided in the prompt) as index.html inside a folder named templates/.

Crucially, ensure that random_forest_regressor_model.pkl (generated from your Jupyter Notebook) and retail_price.csv are in the your-project-folder/ (the same directory as app.py).

2. Install Flask and Other Dependencies
If you haven't already, open your terminal or command prompt and install the necessary Python libraries:

pip install Flask pandas scikit-learn numpy

3. Run the Flask Application
Navigate to the directory where you saved app.py in your terminal or command prompt. Then, run the Flask application using the Python interpreter:

python app.py

You should see output indicating that the Flask development server is running, typically on http://127.0.0.1:5000/.

4. Access the Application
Open your web browser (e.g., Chrome, Firefox, Edge) and go to the following address:

http://127.0.0.1:5000/

You will see a web form where you can input various product features. Upon clicking the "Predict Unit Price" button, the frontend will send this data to your Flask backend. The backend will then use the loaded machine learning model to make a prediction, and the result will be displayed on the web page.
