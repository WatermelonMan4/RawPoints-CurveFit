from os import getenv
from dotenv import load_dotenv
from PolyFitM1 import CurveFitM1
from PolyFitM2 import CurveFitM2
load_dotenv()

if __name__ == '__main__':
    # Creates object of searched function.
    searched_f1 = CurveFitM1(getenv('INPUT_FILE'))
    # Starts fitting procedure.
    searched_f1.fit_run()
    # Plotting results.
    searched_f1.plot_fitted()
    # Overwriting the input file with output data.
    searched_f1.save_fitted()

    searched_f2 = CurveFitM2(getenv('INPUT_FILE'))
    searched_f2.fit_run()
    searched_f2.plot_fitted()
    searched_f2.save_fitted()
