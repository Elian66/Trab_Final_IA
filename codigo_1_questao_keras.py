import numpy as np
from keras.models import Sequential
from keras.layers import Dense


trem1 = np.array([4,4,
		 2,0,2,1,1,
		 3,1,2,1,2,
		 2,0,9,1,4,
		 2,1,2,3,3,
		 1,1,1,0,0,1,0,0
		])

trem2 = np.array([3,3,
		 2,0,1,2,1,
		 2,0,8,1,3,
		 2,0,2,1,4,
		 -1,-1,-1,-1,-1,
		 0,1,0,1,0,0,0,1
		 ])

trem3 = np.array([3,2,
		 3,1,1,1,4,
		 2,0,6,1,4,
		 2,0,2,1,1,
		 -1,-1,-1,-1,-1,
		 0,0,0,0,1,0,1,0
		 ])

trem4 = np.array([4,3,
		 2,0,2,1,3,
		 2,0,4,1,3,
		 2,0,3,1,4,
		 2,0,8,1,4,
		 0,1,0,0,1,0,0,0])

trem5 = np.array([3,3,
		 2,0,1,1,1,
		 3,1,1,1,3,
		 2,0,3,1,4,
		 -1,-1,-1,-1,-1,
		 0,1,0,1,0,0,0,0
		 ])

trem6 = np.array([2,2,
		 2,0,2,1,4,
		 2,1,1,3,1,
		 -1,-1,-1,-1,-1,
		 -1,-1,-1,-1,-1,
		 0,0,0,0,0,0,1,1
		 ])

trem7 = np.array([3,2,
		 2,1,7,0,-1,
		 2,0,2,1,4,
		 2,0,3,1,1,
		 -1,-1,-1,-1,-1,
		 0,0,0,0,0,0,1,0
		 ])

trem8 = np.array([2,2,
		 2,0,2,1,1,
		 3,1,1,1,3,
		 -1,-1,-1,-1,-1,
		 -1,-1,-1,-1,-1,
		 0,0,0,1,0,0,0,0
		 ])

trem9 = np.array([4,2,
		 2,0,8,1,1,
		 2,0,2,1,3,
		 3,1,7,1,3,
		 2,0,8,1,1,
		 1,0,0,1,0,0,0,0
		 ])

trem10 = np.array([2,1,
		  2,1,2,2,3,
		  2,0,2,1,3,
		  -1,-1,-1,-1,-1,
		  -1,-1,-1,-1,-1,
		  1,0,0,0,0,0,0,0
		  ])

entrada = np.stack((trem1,trem2,trem3,trem4,trem5,trem6,trem7,trem8,trem9,trem10))
saida = np.array([1,1,1,1,1,0,0,0,0,0])

model = Sequential()
model.add(Dense(20, input_dim=30, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(entrada, saida, epochs=150, batch_size=10)
# make class predictions with the model
predictions = model.predict_classes(entrada)
# summarize the first 5 cases
for i in range(10):
	print('%s => %d (expected %d)' % (entrada[i].tolist(), predictions[i], saida[i]))
