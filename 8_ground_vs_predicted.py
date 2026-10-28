ground_truth=12
predicted_count=15

print('Absolute Error', abs(predicted_count-ground_truth))

if ground_truth>predicted_count:
    print('Undercount')
elif ground_truth<predicted_count:
    print('Overcount')
else:
    print('counted_exactly')


ground_truth = [8, 12, 5]
predicted = [8, 15, 4]

absolute_error=[]
for i in range(3):
    absolute_error.append(abs(ground_truth[i]-predicted[i]))

import numpy as np
print('Individual Error',absolute_error)
print('MAE', np.mean(absolute_error))