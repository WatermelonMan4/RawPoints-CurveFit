import numpy as np
import pandas as pd
from os import getenv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class CurveFitM1:
    def __init__(self, input_file: str, x=None, y=None):
        self.x = x
        self.y = y
        self.fetch_data(input_file)
        self.popt = None
        self.f_degree = 0
        self.x_new = None
        self.y_new = None

    def fetch_data(self, input_file):
        # load the dataset
        dataframe = pd.read_excel(input_file)
        data = dataframe.values

        # choose the input and output variables
        self.x = data[:, 0]
        self.y = data[:, 1]
        # print imported values
        print("Distance:\n", self.x)
        print("--" * 100)
        print("Height:\n", self.y)
        print("--" * 100)

    @staticmethod
    def model_func(x0, *p):
        poly = 0
        for i, n in enumerate(p):
            poly += n * x0 ** i
        return poly

    def r_squared_func(self, y_line):
        y_avg = np.mean(self.y)
        # SSReg = np.sum((y_line - y_avg) ** 2)
        SSTot = np.sum((self.y - y_avg) ** 2)
        SSErr = np.sum((self.y - y_line) ** 2)
        # r_squared = SSReg / SSTot
        r_squared = 1 - (SSErr/SSTot)

        return round(r_squared, 4)

    def print_formula(self):
        formula = "f(x) = "
        for i, coeff in enumerate(self.popt):
            if i == 0:
                formula += f"{coeff:.2f}"
            else:
                formula += f" + {coeff:.2e}*x^{i}"
        print("\n", "--" * 100, "\n", formula, "\n", "--" * 100)

    def print_stderr(self, pcov):
        p_err = []
        for i in pcov:
            p_err.append(np.sqrt(abs(i[0])))
        for j in range(len(self.popt)):
            print(f"Coeff. {j+1}: {self.popt[j]:.2E} +- {p_err[j]:.2E}")

    def fit_run(self):
        try:
            n = 1
            while True:

                p0 = np.ones(n, )

                self.popt, pcov = curve_fit(self.model_func, self.x, self.y, p0=p0)

                # plot input vs output
                plt.scatter(self.x, self.y, label='Raw data')
                # calculate the output for the range
                y_line = self.model_func(self.x, *self.popt)

                R2 = self.r_squared_func(y_line)

                curve_fit_label = "Polynomial deg. " + str(len(self.popt) - 1) + "\nR2 = " + str(R2)
                plt.plot(self.x, y_line, '--', color='red', label=curve_fit_label)
                plt.xlabel('Distance [m]')
                plt.ylabel('Height change [m]')
                plt.title('Fitting the function to the raw data')
                plt.legend(loc='upper left')

                # print function
                self.print_formula()
                # print standard errors
                self.print_stderr(pcov)

                plt.pause(0.3)

                if R2 >= float(getenv('R_SQUARED')) or n == int(getenv('MAX_ITER')):
                    self.f_degree = len(self.popt) - 1
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
        self.y_new = self.model_func(self.x_new, *self.popt)
        # polynomial function with plotted values
        plt.scatter(self.x, self.y)
        lbl = 'Polynomial deg. ' + str(self.f_degree)
        plt.plot(self.x_new, self.y_new, '-', color='black', label=lbl)
        plt.xlabel('Distance [m]')
        plt.ylabel('Height change [m]')
        plt.title('New values of fitted polynomial function')
        plt.legend(loc='upper left')
        plt.show()

    def save_fitted(self):
        # delta from distance
        delta_x = [0]
        for i in range(1, len(self.x_new)):
            delta_i = round(self.x_new[i] - self.x_new[i - 1], 2)
            delta_x.append(delta_i)
        print('\nDelta x: ', delta_x)
        # delta from height
        delta_y = [0]
        for j in range(1, len(self.y_new)):
            delta_j = round(self.y_new[j] - self.y_new[j - 1], 2)
            delta_y.append(delta_j)
        print('\nDelta y: ', delta_y)
        # overwrite the Excel input data file
        with pd.ExcelWriter(getenv('INPUT_FILE'), engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df_topo = pd.DataFrame({'New dist. [m]': self.x_new, 'Dist. delta [m]': delta_x,
                                    'New height [m]': self.y_new, 'Height delta [m]': delta_y})
            df_topo.to_excel(writer, startcol=2, index=False)
            print("\nThe results were saved to the file:", getenv('INPUT_FILE'))
