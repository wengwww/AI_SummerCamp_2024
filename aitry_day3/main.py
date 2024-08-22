# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# %%
df = pd.read_csv('data/household_power_consumption.txt', sep = ";")
df.head()

# %%
# df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis=1, inplace=True)

# Handle missing values
df.dropna(inplace=True)
# 将所有列的数据类型转换为 float
df = df.astype('float64')

df=df.groupby(df.index // 48).mean()

# %%
#The prediction and test collections are separated over time
# Split training and validation sets
train = df[:int(0.6*len(df))]
valid = df[int(0.6*len(df)):int(0.8*len(df))]
test = df[int(0.8*len(df)):]

# Save the DataFrame to a CSV file
# df.to_csv('data/household_power_consumption_cleaned.csv', index=False)

# print(df.shape)

# Normalization
scaler = MinMaxScaler()
scaler = scaler.fit(train)
train = scaler.transform(train)
valid = scaler.transform(valid)
test =  scaler.transform(test)

# Split X and y
def split_x_and_y(array, days_used_to_train=12):
    features = list()
    labels = list()

    for i in range(days_used_to_train, len(array)):
        features.append(array[i-days_used_to_train:i, :])
        labels.append(array[i, -3])
    return np.array(features), np.array(labels)

train_X, train_y = split_x_and_y(train)
valid_X, valid_y = split_x_and_y(valid)
test_X, test_y = split_x_and_y(test)

# print(train_X.shape, train_y.shape)
# print(valid_X.shape, valid_y.shape)
# print(test_X.shape, test_y.shape)

# %%
# Model establishing and compiling
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=64))
model.add(tf.keras.layers.Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

# %%
# Fitting
model.fit(
    train_X, train_y,
    validation_data=(valid_X, valid_y),
    batch_size=32,
    epochs=50
)

# Predicting
pred_y = model.predict(test_X)
# 检查预测值和真实值的形状
# print("Predicted shape:", pred_y.shape)
# print("Test shape:", test_y.shape)


# 计算均方误差（MSE）
mse = mean_squared_error(test_y, pred_y)
print(f'Mean Squared Error (MSE): {mse}')

# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')
# 绘图

plt.plot(range(len(pred_y)), pred_y, label='Prediction')
plt.plot(range(len(pred_y)), test_y, label='Ground Truth')
plt.xlabel('Amount of samples')
plt.ylabel('Prediction')
plt.legend()
plt.show()

