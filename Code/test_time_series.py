from time_series import TimeSeries

import pandas as pd
import numpy as np


data = pd.read_csv("/Users/graceyin/Downloads/phase1_training_data.csv").to_numpy()
X_all = data[data[:, 0] == "CA"]
y_can = X_all[:, 3]
y_can = np.reshape(y_can, (y_can.shape[0], 1))
y_can = y_can.astype(float)

model = TimeSeries()
X =model.get_tseries_X(y_can, window_length=5, preapp_one=True)
y = model.get_tseries_Y(y_can, window_length=5)
model.fit(X,y)


X1 = np.array((1,9319,9409,9462,9481))

y_pred1 = model.predict(X1)
#print(y_pred1)
X2 = np.array((1,9409,9462,9481,y_pred1))
y_pred2 = model.predict(X2)
#print(y_pred2)
X3 = np.array((1,9462,9481,y_pred1,y_pred2))
y_pred3 = model.predict(X3)
#print(y_pred3)
X4 = np.array((1,9481,y_pred1,y_pred2,y_pred3))
y_pred4 = model.predict(X4)
#print(y_pred4)
X5 = np.array((1,y_pred1,y_pred2,y_pred3,y_pred4))
y_pred5 = model.predict(X5)
#print(y_pred5)
yy=np.array((9578.33175101,9692.15019354,9844.94323193,10081.39684305,10412.25364883))
print(yy)