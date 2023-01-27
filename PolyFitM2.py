import numpy as np
from os import getenv
from dotenv import load_dotenv
from PolyFitM1 import CurveFitM1
import matplotlib.pyplot as plt
load_dotenv()


class CurveFitM2(CurveFitM1):
    def __init__(self, input_file: str, x=None, y=None):
        super().__init__(input_file, x, y)
        self.model_fitted = None

    def fit_run(self):
        try:
            n = 1
            while True:
                fit = np.polyfit(self.x, self.y, n)
                self.model_fitted = np.poly1d(fit)
                # create scatterplot and print function
                polyline = np.linspace(min(self.x), max(self.x), 500)
                y_line = self.model_fitted(self.x)
                print(self.model_fitted)
                plt.scatter(self.x, self.y)

                # add fitted polynomial lines to scatterplot
                R2 = self.r_squared_func(y_line)

                curve_fit_label = "Polynomial deg. " + str(self.model_fitted.order) + "\nR2 = " + str(R2)
                plt.plot(polyline, self.model_fitted(polyline), '--', color='green', label=curve_fit_label)
                plt.xlabel('Distance [m]')
                plt.ylabel('Height change [m]')
                plt.title('Fitting the function to the raw data')
                plt.legend(loc='upper left')

                plt.pause(0.3)

                if R2 >= float(getenv('R_SQUARED')) or n == int(getenv('MAX_ITER')):
                    if n == int(getenv('MAX_ITER')):
                        print('\nPolynomial fit degree limit reached! '
                              '\nNo solution found at R2 >= ' + getenv('R_SQUARED'))
                    break
                plt.clf()
                n += 1
            plt.show()

        except RuntimeError as err:
            print(err)
            exit(100)
        except ValueError as err:
            print(err)
            exit(200)

    def plot_fitted(self):
        self.x_new = np.linspace(min(self.x), max(self.x), int(getenv('NEW_INTERVAL')))
        self.y_new = self.model_fitted(self.x_new)
        # polynomial function with plotted values
        print('\nX-new: ', self.x_new)
        print('Y-new: ', self.model_fitted(self.x_new))
        plt.scatter(self.x, self.y)
        lbl = 'Polynomial deg. ' + str(self.model_fitted.order)
        plt.plot(self.x_new, self.y_new, '-', color='black', label=lbl)
        plt.xlabel('Distance [m]')
        plt.ylabel('Height change [m]')
        plt.title('New values of fitted polynomial function')
        plt.legend(loc='upper left')
        plt.show()
