# Amazon Product Rating Prediction

## 📖 Overview

This project implements a neural network-based recommendation system to predict product ratings on Amazon based on user-product interactions. Using a collaborative filtering approach, the model learns user and product embeddings to predict continuous ratings (1-5 scale). The dataset includes user IDs, product IDs, and ratings from the Amazon review corpus. The project handles class imbalance with oversampling, normalizes ratings, and evaluates using precision-recall metrics. Predictions are generated for the test set and formatted for submission.

The implementation is in a Jupyter Notebook (`what_to_buy.ipynb`) using **Keras/TensorFlow** for the model, **scikit-learn** for preprocessing, and **imbalanced-learn** for oversampling.

## 🎯 Objectives

- **Predict Ratings**: Forecast product ratings (1-5) for unseen user-product pairs.
- **Handle Imbalance**: Address skewed rating distributions (e.g., more high ratings) using RandomOverSampler.
- **Evaluate Performance**: Use precision-recall curves to find the optimal threshold for binary classification (high vs. low ratings).
- **Generate Submission**: Produce a CSV with predicted ratings for the test dataset.

## ✨ Features

- **Data Preprocessing**:
  - Label encoding for users and products.
  - Normalization of ratings to \[0,1\] using MinMaxScaler.
  - Oversampling to balance rating classes.
- **Neural Network Model**:
  - Embedding layers for users and products.
  - Concatenated dense layers with ReLU activation.
  - Output layer for continuous rating prediction.
- **Evaluation**:
  - Precision-Recall curve analysis to determine optimal threshold.
  - Binary classification conversion for high/low rating prediction.
- **Submission Generation**:
  - Inverse normalization and rounding to integer ratings (1-5).
  - Filtered test data based on valid users/items from training.

## 🚀 Usage

1. Open `what_to_buy.ipynb` in Jupyter Notebook.
2. Run cells sequentially:
   - Load and explore training data.
   - Encode users/products and preprocess ratings (handle NaNs, normalize, oversample).
   - Build and train the neural network model.
   - Evaluate with precision-recall and find optimal threshold.
   - Generate predictions for test data and create submission CSV.
3. Output:
   - `submission.csv`: Predicted ratings for test set.
   - Plots: Rating distribution, Precision-Recall curve.

### Key Code Snippets

- **Data Loading & Preprocessing**:

  ```python
  train_data = pd.read_csv('amazon_train.csv')
  train_data = train_data.dropna(subset=['Rating'])
  # Encoding and scaling...
  ```
- **Model Building**:

  ```python
  input_user = Input(shape=(1,))
  input_item = Input(shape=(1,))
  # Embeddings and dense layers...
  model = Model(inputs=[input_user, input_item], outputs=output)
  model.compile(optimizer='adam', loss='mse')
  ```
- **Prediction & Submission**:

  ```python
  y_test_pred = model.predict([X_test_user, X_test_item])
  y_test_pred_rounded = np.clip(np.round(scaler.inverse_transform(y_test_pred)), 1, 5).astype(int)
  submission = pd.DataFrame({'UserID': test_data['UserID'], 'ProductID': test_data['ProductID'], 'Rating': y_test_pred_rounded.flatten()})
  submission.to_csv('submission.csv', index=False)
  ```

## 📊 Code Structure

- **Cell 1**: Load training data and display head.
- **Cell 2**: Label encode users and products.
- **Cell 3**: Handle NaNs, check distribution, normalize ratings, oversample.
- **Cell 4**: Split data, build embedding-based neural network, train model.
- **Cell 5**: Evaluate precision-recall, find optimal threshold.
- **Cell 6**: Predict on test data, inverse transform, round ratings, generate submission.

## 🔍 Evaluation

- **Metrics**: Precision-Recall curve; optimal threshold maximizes F1-score.
- **Imbalance Handling**: RandomOverSampler balances classes (e.g., ratings 1-3 vs. 4-5).
- **Performance**: Model achieves high precision on high ratings; submission focuses on accurate integer predictions.

## 📝 Notes

- Ratings are treated as continuous during training (MSE loss) but rounded to integers for submission.
- Test data is filtered to include only users/products seen in training to avoid cold-start issues.
- The model uses user/product embeddings (size 50) for collaborative filtering.
- Potential Improvements: Hyperparameter tuning, early stopping, or advanced architectures like autoencoders.