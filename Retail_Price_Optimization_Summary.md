# **Retail Price Optimization Project Summary**

This document outlines a machine learning project focused on developing an optimized pricing strategy for the retail industry. The project encompasses data analysis, predictive model development, and the deployment of a web-based application for practical implementation.



#### 1\. Problem Definition

The retail sector operates within a highly competitive landscape where pricing significantly influences customer attraction and profit maximization. Pricing decisions are inherently complex, being influenced by a multitude of factors including, but not limited to, cost of goods sold (COGS), prevailing market demand, competitive pricing strategies, and desired profit margins. The primary objective of this project is to implement a data-driven methodology to ascertain optimal pricing points that simultaneously enhance sales volume and sustain profitability.



#### 2\. Data Source and Description

The analysis was conducted using the retail\_price.csv dataset, comprising 676 records and 30 distinct features. Key attributes within the dataset include:



product\_id: A unique identifier for each product.



product\_category\_name: A categorical descriptor of the product type.



month\_year: The temporal reference for each data entry.



qty: The quantity of units sold.



total\_price: The aggregate revenue generated from sales.



freight\_price: The cost associated with product shipping.



unit\_price: The price per individual unit, designated as the primary target variable for prediction.



Product-specific attributes: product\_name\_lenght, product\_description\_lenght, product\_photos\_qty, product\_weight\_g, and product\_score (customer review rating).



Customer and Temporal Metrics: customers (number of customers), weekday, weekend, holiday, month, and year (temporal indicators).



Market Dynamics: s (likely a seasonality or sales index), and volume (product sales volume).



Competitive Intelligence: comp\_1, comp\_2, comp\_3 (competitor pricing), ps1, ps2, ps3 (competitor product scores), and fp1, fp2, fp3 (competitor freight prices).



lag\_price: The unit price from the preceding period, identified as a highly influential factor.







#### 3\. Data Preprocessing

To prepare the raw data for machine learning model ingestion, the following preprocessing steps were executed:



Date Transformation: The month\_year column, initially stored as a string, was converted into a datetime object to facilitate temporal analysis.



Feature Exclusion: The product\_id column was removed due to its role as a unique identifier, which does not contribute to the model's generalizability for pricing across diverse products. The total\_price column was also excluded to prevent data leakage, as it is a direct derivative of the target variable (unit\_price) and qty.



Categorical Encoding: The product\_category\_name feature, being categorical, was transformed into a numerical representation using one-hot encoding. This process generated binary indicator columns (e.g., product\_category\_name\_computers\_accessories), with drop\_first=True applied to mitigate multicollinearity.



Target Variable Assignment: The unit\_price column was explicitly designated as the dependent variable (y) for the regression task, with all other relevant, preprocessed columns serving as independent features (X).





#### 4\. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to gain insights into the dataset's characteristics, distributions, and inter-feature relationships:



Target Variable Distribution: A histogram, augmented with a Kernel Density Estimate (KDE), was generated to visualize the distribution of unit\_price, revealing common price ranges and distributional properties.



Inter-feature Correlation: A correlation heatmap was produced for numerical features to identify the strength and direction of relationships between variables, particularly those influencing unit\_price.



Outlier Identification: Box plots were utilized to visually inspect the distributions of key numerical features such (e.g., qty, freight\_price, product\_weight\_g, customers), aiding in the identification of potential outliers.





#### 5\. Model Selection and Training

The core of the project involved selecting, training, and validating a machine learning model:



Model Selection: A RandomForestRegressor was chosen for this regression problem. This ensemble learning method is recognized for its robustness, high predictive accuracy, capacity to handle non-linear relationships, and inherent resistance to overfitting.



Data Partitioning: The preprocessed dataset was systematically divided into training (80%) and testing (20%) subsets using train\_test\_split. This rigorous separation ensures an unbiased evaluation of the model's performance on unseen data.



Model Training: The RandomForestRegressor model was trained on the X\_train and y\_train datasets, enabling it to learn the underlying patterns and relationships within the data.





#### 6\. Model Evaluation

The trained model's efficacy was quantitatively assessed using standard regression metrics on the test set:



Mean Absolute Error (MAE): Provides the average absolute difference between predicted and actual values.



Mean Squared Error (MSE): Calculates the average of the squared differences, penalizing larger errors more significantly.



Root Mean Squared Error (RMSE): The square root of the MSE, expressing the error in the same units as the target variable.



R-squared (R 

2

&nbsp;): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.



The model achieved an R-squared value of approximately 0.93, signifying that approximately 93% of the variance in unit\_price can be accounted for by the model's features, indicating strong predictive capability.



#### 7\. Feature Importance Analysis

An analysis of feature importance was conducted to identify the most influential factors contributing to the model's predictions:



lag\_price: This feature demonstrated the highest importance (approximately 81%), underscoring the significant impact of historical pricing on current unit prices.



Other notable features influencing the predictions included comp\_1 (competitor 1 price), freight\_price, product\_weight\_g, and volume.





#### 8\. Web Application Deployment

To facilitate practical utilization of the trained model, a web application was developed, comprising a Flask backend and an HTML/JavaScript frontend:



Model Persistence: The trained RandomForestRegressor model was serialized and saved as random\_forest\_regressor\_model.pkl using Python's pickle module, allowing for efficient loading and inference without repeated training.



Flask Backend (app.py):



Responsible for loading the random\_forest\_regressor\_model.pkl and retail\_price.csv (for dynamic population of product categories).



Implements a /predict API endpoint that processes incoming JSON payloads containing product features.



Performs necessary preprocessing on the input data to align with the model's expected feature format, including the application of one-hot encoding for categorical variables.



Generates a unit\_price prediction using the loaded model.



Transmits the prediction back to the frontend as a JSON response.



Frontend (index.html):



Presents a user-friendly web form for inputting product attributes.



Leverages Tailwind CSS for a modern, responsive, and business-themed aesthetic.



Integrates Lucide Icons to provide intuitive visual cues for input fields.



Utilizes JavaScript to:



Collect user inputs from the form.



Dispatch asynchronous POST requests to the Flask /predict API.



Dynamically display the predicted unit\_price.



Incorporate smooth opening and closing transitions for loading indicators and prediction results, enhancing user experience.



Features a dark/light mode toggle for theme customization, with user preferences persisted via local storage.

