import numpy as np 

def gradient_descent(x , y):
    cost_F = {}
    m_slope = b_intercept = 0 
    m = len(x)
    epochs = 1000
    learning_rate = .01

    for i in range(1 , epochs + 1 ):

        # Predictions:
        y_pred = m_slope * x + b_intercept 
        
        # Gradient Descent:
        dm = (1/m) * np.sum(y_pred - y)
        db = (1/m) * np.sum((y_pred - y) * x)
        
        # Update Parameters:
        m_slope = m_slope - learning_rate * dm
        b_intercept = b_intercept - learning_rate * db

        # Cost Function: 
        cost = (1/2*m) * np.sum((y_pred - y)**2)
        cost_F[i] = cost
        print("m {}, b {}, cost {} iteration {}".format(m_slope, b_intercept, cost, i)) 

    print(f"{"=" * 50}\nMax Cost Function:", max(cost_F.values()) , " iter: ", max(cost_F, key=cost_F.get) )
    print("min Cost Function:", min(cost_F.values()) , " iter: ", min(cost_F, key=cost_F.get) )    
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x, y)