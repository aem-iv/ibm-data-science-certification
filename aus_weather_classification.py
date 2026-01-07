"""
Final Project: AUS Weather Classification

Author: aem-iv
Date: January 2026
"""

from __future__ import annotations


def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import seaborn as sns

    # Load data
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
    df = pd.read_csv(url)

    # Clean data
    df = df.dropna()

    df = df.rename(columns={
        'RainToday': 'RainYesterday',
        'RainTomorrow': 'RainToday'
    })

    df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia'])]

    # Feature engineering: Date → Season
    def date_to_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'

    df['Date'] = pd.to_datetime(df['Date'])
    df['Season'] = df['Date'].apply(date_to_season)
    df = df.drop(columns='Date')

    # Split features/target
    X = df.drop(columns='RainToday', axis=1)
    y = df['RainToday']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Identify feature types
    numeric_features = X_train.select_dtypes(
        include=['number']).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=['object', 'category']).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # --------------------
    # Random Forest Model
    # --------------------
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    # ✅ FIXED: added random_state for reproducibility
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    print(
        "Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    test_score = grid_search.score(X_test, y_test)
    print("Test set score: {:.2f}".format(test_score))

    y_pred = grid_search.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(conf_matrix).plot(cmap='Blues')
    plt.title("Random Forest Confusion Matrix")
    plt.show()

    # Feature importance
    feature_importances = grid_search.best_estimator_[
        'classifier'].feature_importances_

    feature_names = (
        numeric_features +
        list(
            grid_search.best_estimator_
            ['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(categorical_features)
        )
    )

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    top_features = importance_df.head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Most Important Features for Rain Prediction")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    # --------------------
    # Logistic Regression
    # --------------------
    pipeline.set_params(
        classifier=LogisticRegression(random_state=42)
    )

    grid_search.estimator = pipeline

    param_grid = {
        'classifier__solver': ['liblinear'],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__class_weight': [None, 'balanced']
    }

    grid_search.param_grid = param_grid
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
