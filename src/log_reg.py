import numpy as np

class LogisticReg():

    n_iters = 25000
    learning_rate = 0.01

    def __init__(self) -> None:
        pass

    def _normalize(self) -> None:
        mu = np.mean(self.X,1).reshape(self.p,1)
        sigma = np.std(self.X,1).reshape(self.p,1)
        self.X = (self.X - mu) / sigma

    def _initialize_theta_vector(self):
        self.theta_vector = np.zeros((self.p,1), dtype="float")

    def _initialize_bias(self):
        self.bias = 0.0
        
    def _sigmoid(self,vect):
        self.sig_vals = 1 / (1+np.exp(-vect))    
        return self.sig_vals

    def _calculate_linear_combination(self):
        #print("Theta_T shape", self.theta_vector.T.shape)
        #assert self.theta_vector.T.shape[1] == self.X.shape[0], "Dimensions don't match"
        self.z = np.dot(self.theta_vector.T,self.X)

    def _compute_cost(self):
        self._calculate_linear_combination()
        self._sigmoid(self.z)

        J = (-1/self.m) * np.sum( (self.y*np.log(self.sig_vals)) + ((1-self.y)*np.log(1-self.sig_vals)) )
        return J

    def _compute_gradient_step_values(self):
        self.d_theta_vect = (1/self.m)*np.dot(self.X,(self.sig_vals-self.y).T)
        self.d_bias = (1/self.m)*np.sum(self.sig_vals-self.y)

    def _execute_gradient_descent_step(self):
        self.theta_vector = self.theta_vector - (self.learning_rate*self.d_theta_vect)
        self.bias = self.bias - (self.learning_rate*self.d_bias)

    def fit(self,X,y):
        self.X = np.array(X).T
        self.y = np.array(y).T
        self.m = self.X.shape[1]
        self.p = self.X.shape[0]


        self._normalize()
        self._initialize_theta_vector()
        self._initialize_bias()

        for _ in range(self.n_iters):
            cost_t0 = self._compute_cost()
            self._compute_gradient_step_values()
            self._execute_gradient_descent_step()
            cost_t1 = self._compute_cost()

        self.tuned_theta_vector = self.theta_vector
        self.y_preds = self.tuned_theta_vector.T @ self.X
        self._sigmoid(self.y_preds)
        self.y_class = self.sig_vals >= 0.5
        self.score = np.sum(self.y == self.y_class) / self.m


        

    
