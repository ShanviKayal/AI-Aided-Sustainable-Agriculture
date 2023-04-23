import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train():
    # df = pd.read_csv("SBP.csv")

    # x = df[["Age", "Weight"]]
    # y = df["SBP"]

    # regr = LinearRegression()
    # regr.fit(x, y)

    # joblib.dump(regr, "regr.pkl")
    # Step 1: Collect the data
# Assume that you have collected the data and stored it in a CSV file called 'crop_data.csv'


# Step 2: Preprocess the data
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    from sklearn.ensemble import RandomForestRegressor

    crop_data = pd.read_csv('yield.csv')
    crop_data = crop_data.dropna()
    X = crop_data[['Pesticide_use', 'Avg_rainfall', 'Avg_temperature','Area_used']]
    y = crop_data['Crop_yield']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate the mean squared error and R-squared
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # print(f"Mean Squared Error: {mse}")
    # print(f"R-squared: {r2}")
    # new_data = pd.DataFrame({'Pesticide_use': [2.5], 'Avg_rainfall': [100], 'Avg_temperature': [30],'Area_used' : [100] })
    # new_data_scaled = scaler.transform(new_data)
    # yield_pred = model.predict(new_data_scaled)
    # model.save("model1.h5")
    # print(f"Predicted yield: {yield_pred}")
    joblib.dump(model, "regr1.pkl")


def load():
    clf = joblib.load("regr.pkl")
    age = 18
    weight = 60
    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    prediction = clf.predict(x)[0]
    print(prediction)


if __name__ == "__main__":
    train()
    load()
