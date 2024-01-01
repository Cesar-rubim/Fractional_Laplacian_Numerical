import numpy as np 
from scipy.special import roots_hermite, gamma, hyp1f1
from scipy.integrate import quad
from math import sqrt, factorial, e

def get_f(y, x_i):
    return (e**(-x_i**2) -  e**(-y**2))/(abs(x_i - y)**3)


class DMatrix:

    def __init__(self, alpha, order):
        self.alpha = alpha
        self.n = order

    def get_hermite_roots(self) -> None:
        """Get hermite roots of the Nth Hermite polynomial"""
        self.x = roots_hermite(self.n, mu=False)[0]

    def get_components(self):
        D_matrix = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            for j in range(0,self.n):
                # print(j)
                if (j)%2==0: #even j
                    n = j/2
                    D_matrix[i][j] = 2**(self.alpha)*(-1)**(n)*sqrt(factorial(2*n))/(2**(n)*factorial(n)) \
                                        *gamma(n + self.alpha/2+1/2)/(gamma(n + 1/2)) \
                                            *hyp1f1(n + self.alpha/2+1/2, 1/2,-(self.x[i]**2))
                else:
                    n = (j-1)/2
                    D_matrix[i][j] = 2**(self.alpha+1)*(-1)**(n)*sqrt(factorial(2*n+1))/(2**(n+1/2)*factorial(n)) \
                                        *gamma(n + self.alpha/2+3/2)/(gamma(n + 3/2)) \
                                            *self.x[i]*hyp1f1(n + self.alpha/2+3/2, 3/2,-(self.x[i]**2))

        return D_matrix
    


    def get_rho_matrix(self):
        rho_matrix = np.zeros(self.n)
        for i in range(self.n):
            rho_matrix[i] = 0
            # if i%2 == 0:
            #     rho_matrix[i] = 1
            # else:
            #     rho_matrix[i] = 0
        return rho_matrix
    


    def get_f_matrix(self):
        f_matrix = np.zeros(self.n)
        for i in range(self.n):
            f_matrix[i] = 0 #quad(get_f, -np.inf, np.inf, limit=100,args=(self.x[i]))[0]/100000
            # print(f_matrix[i])
            f_matrix[i] = 2**(self.alpha)*(-1)*sqrt(2)/(2) \
                                        *gamma(self.alpha/2+3/2)/(gamma(3/2)) \
                                            *hyp1f1(self.alpha/2+3/2, 1/2,-(self.x[i]**2))
            # f_matrix[i] = e**((self.x[i]**2)/2)*(1+self.x[i])
            
        return f_matrix

    
    def solver(self, A, B):
        print(A)
        print(B)
        return np.linalg.solve(A,B)


    def _execute(self):
        self.get_hermite_roots()
        D = self.get_components()
        rho = self.get_rho_matrix()
        F = self.get_f_matrix()

        return self.solver(A=D+rho, B=F)




    
if __name__=="__main__":
    N = 10
    alpha = 0.5
    T = DMatrix(alpha=alpha, order=N)._execute()
    print(T)


    
    