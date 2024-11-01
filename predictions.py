import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


mlflow.set_tracking_uri('https://dagshub.com/GOLISHYAMP/MLFlow.mlflow')

# Load model as a PyFunc.
logged_model = 'runs:/d3203620589a42da9ca259a4933b80c9/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)


# Read the wine-quality csv file from the URL
csv_url = (
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
)
try:
    data = pd.read_csv(csv_url, sep=";")
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]


# Predict on a pandas DataFrame.
y_pred = loaded_model.predict(test_x)
print(y_pred)
