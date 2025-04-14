import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def balance_dataset(input_path="data/cleaned_combined_dataset.csv", output_path="data/balanced_dataset.csv"):
    # Load the cleaned dataset
    df = pd.read_csv(input_path)

    # Drop non-numeric or unwanted columns
    columns_to_drop = [col for col in ['Time', 'Timestamp'] if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)

    # Display initial class distribution
    print("Before Balancing:")
    print(df["State"].value_counts())

    # Define features and label
    X = df.drop("State", axis=1)
    y = df["State"]

    # Function to apply jitter for augmentation
    def jitter_data(X, noise_level=0.05):
        noise = np.random.normal(loc=0, scale=noise_level, size=X.shape)
        return X + noise

    # Augment the minority class using jitter
    minority_df = df[df['State'] == 1]
    minority_jittered = minority_df.copy()
    minority_jittered[X.columns] = jitter_data(minority_df[X.columns], noise_level=0.1)

    # Combine original and augmented data
    augmented_df = pd.concat([df, minority_jittered], ignore_index=True)

    # Separate features and labels again
    X_aug = augmented_df.drop("State", axis=1)
    y_aug = augmented_df["State"]

    # Under-sample majority class using RandomUnderSampler with a target ratio
    rus = RandomUnderSampler(sampling_strategy=0.9, random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X_aug, y_aug)

    # Display final class distribution
    print("\nAfter Augmentation and Balancing:")
    print(pd.Series(y_balanced).value_counts())

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Train a simple model for evaluation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    print("\nModel Evaluation:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Save the final balanced dataset
    balanced_data = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name="State")], axis=1)
    balanced_data.to_csv(output_path, index=False)
    print(f"\nBalanced dataset saved as '{output_path}'")

    return output_path
