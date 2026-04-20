import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_data(df):

    # Remove unnecessary column
    df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    # Encode categorical columns
    encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = encoder.fit_transform(df[column])

    return df


# ----------------------------
# Optional Testing Block
# ----------------------------
if __name__ == "__main__":

    df = pd.read_csv("../dataset/telco_customer_churn.csv")
    df = preprocess_data(df)

    print("Preprocessing Completed")
    print(df.head())