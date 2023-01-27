import numpy as np
import pandas as pd
from os import getenv
from dotenv import load_dotenv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
load_dotenv()


def objective_polynomial(x_inp, *p):
    poly = 0
    print("--")
    for i, n in enumerate(p):
        poly += n * x_inp ** i
        #print(f"Poly {i}: {x_inp}")
    return poly


def r_squared_func(y_fit):
    y_avg = np.mean(y)
    # SSReg = np.sum((y_line - y_avg) ** 2)
    SSTot = np.sum((y - y_avg) ** 2)
    SSErr = np.sum((y - y_fit) ** 2)
    # r_squared = SSReg / SSTot
    r_squared = 1 - (SSErr/SSTot)

    return round(r_squared, 4)


def print_formula(p):
    formula = "f(x) = "
    for i, factor in enumerate(p):
        if i == 0:
            formula += f"{factor:.2f}"
        else:
            formula += f" + {factor:.2e}*x^{i}"
    print("\n", "--" * 100, "\n", formula, "\n", "--" * 100)


def print_stderr(coeff, covar):
    p_err = []
    for i in covar:
        p_err.append(np.sqrt(abs(i[0])))
    for j in range(len(coeff)):
        print(f"Wsp. {j+1}: {coeff[j]:.2E} +- {p_err[j]:.2E}")


def fit_loop(x_in, y_in):
    try:
        n = 1
        while True:

            p0 = np.ones(n, )

            param, pcov = curve_fit(objective_polynomial, x_in, y_in, p0=p0)
            print("---Koniec 'curve_fit'---")
            # plot input vs output
            plt.scatter(x_in, y_in, label='Dane surowe')
            # calculate the output for the range
            y_line = objective_polynomial(x_in, *param)
            print("---Koniec y_line---")
            R2 = r_squared_func(y_line)

            curve_fit_label = "Wielomian st. " + str(len(param) - 1) + "\nR2 = " + str(R2)
            plt.plot(x_in, y_line, '--', color='red', label=curve_fit_label)
            plt.xlabel('Odległość [m]')
            plt.ylabel('Zmiana wysokości [m]')
            plt.title('Dopasowanie funkcji do danych surowych')
            plt.legend(loc='upper left')

            # print function
            print_formula(param)
            # print standard errors
            print_stderr(param, pcov)
            plt.pause(0.3)

            if R2 >= float(getenv('R_SQUARED')) or n == int(getenv('MAX_ITER')):
                f_deg = len(param) - 1
                if n == int(getenv('MAX_ITER')):
                    print('\nOsiągnięto limit stopnia dopasowania wielomianu. '
                          '\nNie znaleziono rozwiązania na poziomie R2 >= ' + getenv('R_SQUARED'))
                break

            plt.clf()
            n += 1
        plt.show()
        return param, f_deg
    except RuntimeError as err:
        print(err)
        exit(100)
    except ValueError as err:
        print(err)
        exit(200)


def plot_fitted(x_org, y_org):
    x_interval = np.linspace(min(x_org), max(x_org), int(getenv('NEW_INTERVAL')))
    # result polynomial function with plotted values
    plt.scatter(x_org, y_org)
    lbl = 'Wielomian st. ' + str(f_degree)
    plt.plot(x_interval, objective_polynomial(x_interval, *popt), '-', color='black', label=lbl)
    plt.xlabel('Odległość [m]')
    plt.ylabel('Zmiana wysokości [m]')
    plt.title('Nowe wartości dopasowanego wielomianu ')
    plt.legend(loc='upper left')
    plt.show()
    return x_interval


def save_fitted(x_out, y_out):
    # delta z odległości
    delta_x = [0]
    for i in range(1, len(x_out)):
        delta_i = round(x_out[i] - x_out[i-1], 2)
        delta_x.append(delta_i)
    print('\nDelta x: ', delta_x)
    # delta z wysokośći
    delta_y = [0]
    for j in range(1, len(y_out)):
        delta_j = round(y_out[j] - y_out[j-1], 2)
        delta_y.append(delta_j)
    print('\nDelta y: ', delta_y)
    # overwrite the excel input data file
    with pd.ExcelWriter(getenv('INPUT_FILE'), engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        df_topo = pd.DataFrame({'Nowe odc. [m]': x_out, 'Delta odc. [m]': delta_x,
                                'Nowe wys. [m]': y_out, 'Delta wys. [m]': delta_y})
        df_topo.to_excel(writer, startcol=2, index=False)
        print("Poprawnie zapisano wyniki do pliku:", getenv('INPUT_FILE'))


if __name__ == '__main__':
    # load the dataset
    dataframe = pd.read_excel(getenv('INPUT_FILE'))
    data = dataframe.values

    # choose the input and output variables
    x, y = data[:, 0], data[:, 1]

    print("Odległość:\n", x)
    print("--" * 100)
    print("Wysokość:\n", y)
    print("--" * 100)

    # curve fit loop
    popt, f_degree = fit_loop(x, y)

    # plot fitted function
    x_new = plot_fitted(x, y)
    y_new = objective_polynomial(x_new, *popt)
    y_new = np.array(y_new)
    print('\nx-new', x_new)
    print('\ny-new', y_new)
    # save output
    save_fitted(x_new, y_new)
