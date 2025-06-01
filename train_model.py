import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Pisahkan fitur dan label
X = df.drop(columns='diabetes')
y = df['diabetes']

# Kolom numerik dan kategorik
numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformasi data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Buat model ANN
model = Sequential([
    Dense(16, input_shape=(X_train_transformed.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callback early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Latih model
model.fit(X_train_transformed, y_train, validation_split=0.2, epochs=20, callbacks=[early_stop])

# Simpan model dan preprocessor
model.save("model/ann_diabetes_model.h5")
joblib.dump(preprocessor, "model/preprocessor.pkl")
