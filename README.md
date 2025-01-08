# Sales Forecasting System

This project implements a sales forecasting system using time series analysis to predict retail sales for the next quarter. It utilizes models such as SARIMA and Facebook Prophet, along with data preprocessing, feature engineering, a Flask API, and Docker for containerization.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Technologies Used](#technologies-used)
4.  [Project Structure](#project-structure)
5.  [Setup Instructions](#setup-instructions)
6.  [Usage](#usage)
7.  [API Endpoints](#api-endpoints)
8.  [Docker](#docker)
9.  [Testing](#testing)
10. [Model Evaluation](#model-evaluation)
11. [Future Improvements](#future-improvements)
12. [License](#license)
13. [Contact](#contact)

## Introduction

This project demonstrates how to build a complete sales forecasting system by:

*   Preprocessing historical sales data.
*   Extracting relevant time-based and lagged features.
*   Implementing SARIMA and Facebook Prophet models to capture trends and seasonality.
*   Providing a Flask API for easy access to predictions.
*   Containerizing the application for easy deployment using Docker.

## Features

*   **Time Series Analysis:** Uses time series models for forecasting.
*   **Data Preprocessing:** Handles missing values and outliers in sales data.
*   **Feature Engineering:** Creates lagged features, and time-based features from data.
*   **Multiple Models:** Implements and uses both SARIMA and Prophet models.
*   **External Variables:** Integrates external variables to improve predictive power (e.g. temperature, holidays)
*   **Flask API:** Provides an API endpoint for making predictions.
*   **Dockerized Application:** Containerized using Docker for easy deployment.
*   **Model Evaluation:** Includes evaluation metrics (MAE, RMSE) for assessing model performance.

## Technologies Used

*   **Python:** Primary programming language.
*   **Pandas:** For data manipulation and analysis.
*   **Statsmodels:** For implementing the SARIMA model.
*   **Prophet:** Facebook's time-series forecasting library.
*   **Scikit-learn:** For evaluation metrics and train test splitting.
*   **Flask:** A micro web framework for building the API.
*   **Docker:** For containerization and deployment.
*   **Git LFS:** For managing large files.
* **Matplotlib**: For visualizations

## Project Structure

sales-forecasting-system/
├── data/ # CSV, Excel files
├── notebooks/ # Exploratory Analysis and Model Development
├── src/ # Python code
│ ├── data_preprocessing.py # Handles data loading, cleaning and merging
│ ├── feature_engineering.py # Creates the lagged and time based features
│ ├── models.py # Model training and prediction
│ ├── app.py # Flask application
├── tests/
│ ├── test_data_processing.py
│ ├── test_models.py
├── requirements.txt # List of Python dependencies
├── Dockerfile # For Dockerization
├── docker-compose.yml # For Docker Compose (optional, but good)
└── README.md # Documentation

## Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/vyshnev/sales-forecasting-system.git
    ```

2.  **Navigate to the Project Directory:**

    ```bash
    cd sales-forecasting-system
    ```

3.  **Create a Virtual Environment:**

        ```bash
        python -m venv venv
        source venv/bin/activate # For Linux/macOS
        .\venv\Scripts\activate # For Windows
        ```

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5. **Install Git LFS:**
        ```bash
        git lfs install
        ```

6.  **Install Docker Desktop:**
   * Download and install Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

## Usage

1.  **Build and Run with Docker Compose:**

    ```bash
    docker compose up --build
    ```
    This command builds the docker image, and starts the container.

2.  **Access the API:**

    *   The API will be running on `http://localhost:5000`.
3.  **Test the API**
    *   Use a tool like Postman or `curl` to send POST requests to `http://localhost:5000/predict` using an empty JSON object as a request body.

## API Endpoints

*   **`/predict` (POST):**
    *   Accepts a JSON object in the request body.
    *   Returns a JSON response with future sales forecasts for both the SARIMA and the Prophet model.

## Docker

This project is containerized using Docker. The `Dockerfile` in the root directory specifies the steps to build the image. We are using `docker-compose.yml` to define the image creation and the port forwarding.
*   **`docker compose up --build`**: builds the docker image and starts the container.

## Testing

* The project contains unit tests in the `tests` directory, to verify basic functionality.

## Model Evaluation

*   The SARIMA and Prophet models' performance is evaluated using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error). These metrics are shown in the output, when the `data_exploration.ipynb` (or `data_exploration.py`) files are run.
*   A visual representation of the results can also be found in the output, where it is plotted using matplotlib.
*   **Future predictions** from both the models are returned in the API response.

## Future Improvements

*   **Hyperparameter Tuning:** Implement hyperparameter tuning to improve model performance.
*   **More Advanced Models:** Try more advanced models like LSTM, or other ensemble models.
*   **Additional Regressors:** Include more external regressors to improve model accuracy.
*  **Robust API**: Implement a more robust API to handle different types of requests.
*   **Monitoring and Logging**: Implement application logging and metrics gathering.
*   **Multi-Stage Builds**: Use multi stage builds in docker to reduce the size of the container.
*   **Automated Deployment**: Deploy the application to the cloud.

## License

This project is licensed under the MIT License.