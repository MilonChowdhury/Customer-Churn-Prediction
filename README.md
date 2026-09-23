# Customer Churn Prediction

A Streamlit web application that predicts whether a customer is likely to churn using a trained TensorFlow/Keras neural network.

## Project Files

- `app.py` - Streamlit prediction application
- `experiments.ipynb` - Data preparation, model training, evaluation, and TensorBoard experiments
- `churn_test_data.csv` - Customer dataset used for training and testing
- `churn_model.h5` - Saved trained Keras model
- `x_train_scaler.pkl` - Fitted feature scaler
- `*_encoder.pkl` - Saved categorical encoders
- `logs/` - TensorBoard training logs
- `req.txt` - Core Python dependencies

## Requirements

- Python 3.12 or a compatible Python version
- TensorFlow
- Streamlit
- pandas
- NumPy
- scikit-learn

## Setup

From the project directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r req.txt
pip install streamlit
```

## Run the Application

Start the Streamlit app from the project directory:

```bash
streamlit run app.py
```

Open the URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Enter the customer's details and select **Predict Churn**. The app displays the churn probability and predicts whether the customer is likely to churn or stay.

## Model Input Features

The model uses:

- Tenure, monthly charges, total charges, and support tickets
- Senior citizen, partner, dependents, online security, tech support, and streaming status
- Contract type
- Payment method
- Internet service type

Categorical values are transformed using the saved encoders, then scaled using `x_train_scaler.pkl` before being passed to the model.

## Notebook Workflow

Run the notebook cells in order to reproduce the preprocessing and training workflow:

1. Load the CSV data.
2. Encode categorical columns.
3. Create independent features `X` and target `y`.
4. Split and scale the data.
5. Train the neural network.
6. Save the model, scaler, and encoders.
7. Use TensorBoard to inspect training logs.

To view TensorBoard from the notebook:

```python
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

## Notes

Run `streamlit run app.py` from the project root so the model and `.pkl` files can be found using their relative paths.
