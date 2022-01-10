
# RUN on CMD
# >>>python testing.py 200 400 12 # sys.argv -> x y epochs

from keras.models import load_model
import numpy as np, sys, os
import matplotlib.pyplot as plt

try:
    model_file = f'./model/model_{sys.argv[3]}.h5'
    x, y = float(sys.argv[1]), float(sys.argv[2])

except:
    model_file = './model/model_12.h5' # default
    x, y = 100.0, 200.0

model = load_model(model_file)

a= np.array([[x,y]])
z = model.predict(a)
z = float(z[0][0])

# os.system('CLS')
print('*'*50, end='\n\n')
print(f'{x} + {y} = {z}')
print()

# ----------------------------------

x_, y_ = [], []
for i in range(100):
    x, y = i, i+1
    x_.append(x)
    y_.append(y)

    a= np.array([[x,y]])
    z = model.predict(a)

    z = float(z[0][0])
    print(f'{x} + {y} = {z}')

plt.scatter(x_, y_)
plt.show()

# ------------------------------

# import sys
# print("This is the name of the script:", sys.argv[0])
# print("Number of arguments:", len(sys.argv))
# print("The arguments are:" , str(sys.argv))
#
# print()
# print(sys.argv[9]) # IndexError: list index out of range
