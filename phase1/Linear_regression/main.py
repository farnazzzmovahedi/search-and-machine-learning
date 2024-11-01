import numpy as np
from sgd.sgd import SGD
from sgd.utils import load_data, preprocess_data, normalize_features
from sgd.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    train_df = load_data('data/train.csv')
    test_df = load_data('data/test.csv')
    X_train, y_train = preprocess_data(train_df)
    X_test, y_test = preprocess_data(test_df)
    X_train = normalize_features(X_train)
    X_test = normalize_features(X_test)

    model = SGD(lr=0.001, epochs=100, batch_size=512, tol=1e-3, momentum=0.9, l2_lambda=0.001, decay_rate=0.95)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_loss = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test Loss (MSE): {test_loss}, R² Score: {r2}", f"Test Mean Absolute Error (MAE): {test_mae}")

if __name__ == '__main__':
    main()
