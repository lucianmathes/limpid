import numpy as np

from scipy.optimize import least_squares as least_squares_scipy
from scipy.linalg import svd

# Wrapper for scipy's least_squares. Inspired by lmfit.


class FitParameter:
    """
    The FitParameter class contains the values associated with one parameter.

    ...

    Parameters
    ----------
    name : str
        Name of the parameter.
    value : float
        Value of the parameter. Default None.
    vary : bool
        Shows if parameter was fitted. Default True.
    min : float
        Lower bound. Default -inf.
    max : float
        Upper bound. Default inf.
    """

    def __init__(self, name, value=None, vary=True, min=-np.inf, max=np.inf):
        self.name = name
        self.value = value
        self.vary = vary
        self.min = min
        self.max = max
        self.precision = 8  # floating point precision in digits, used for rounding in the terminal print-out.

    def __str__(self):
        """
        Default string implementation.

        Returns
        -------
        string
        """

        final = self.name + ": " + str(round(self.value, self.precision)) \
                + ", vary: " + str(self.vary) \
                + ", min: " + str(self.min) + ", max: " + str(self.max)

        return final


class ResultParameter:
    """
    ResultParameter class contains the results associated with one parameter.

    ...

    Parameters
    ----------
    name : str
        Name of the parameter.
    value : float
        Value of the parameter.
    vary : bool
        Shows if parameter was fitted.
    mn : float
        Lower bound.
    mx : float
        Upper bound.
    init : float
        Initial value used in fitting procedure.
    stderr : float
        stderr from covariance matrix.
    precision : int
        Floating Point precision (in digits). Used for rounding in print-out.
    """

    def __init__(self, name, value, vary, mn, mx, init, stderr, precision=8):
        self.name = name
        self.value = value
        self.vary = vary
        self.min = mn
        self.max = mx
        self.init = init
        self.stderr = stderr
        self.precision = precision

    def __str__(self):
        """
        Default string implementation.

        Returns
        -------
        string
        """
        fixed_vary_str = ", fixed"
        if self.vary:
            fixed_vary_str = ""

        final = self.name + ": " + str(round(self.value, self.precision)) + " +/- " \
                + str(round(self.stderr, self.precision)) \
                + " ( init: " + str(round(self.init, self.precision)) \
                + fixed_vary_str + " )"

        return final


class FitParameters:
    """
    The FitParameters class contains all necessary FitParameter objects and provides all methods for manipulation.
    """

    def __init__(self):
        self.dict = {}  # dict containing all FitParameter objects
        self.str = "Fit Parameters:\n"

    def __iter__(self):
        """
        Class iterator.

        Returns
        -------

        """
        return iter(self.dict)

    def __getitem__(self, name):
        """
        Returns FitParameter.

        Parameters
        ----------
        name : str
            Name of the FitParameter

        Returns
        -------
        FitParameter
        """

        return self.dict[name]

    def __str__(self):
        """
        Default string implementation, which returns a structured string.

        Returns
        -------
        string
        """

        def add_line(line):
            """
            Add a line to the final string with a line break at the end.
            Parameters
            ----------
            line : str
                Line to be added

            Returns
            -------

            """
            self.str += line + "\n"

        add_line("-" * len(self.str))

        for parameter in self.dict:
            add_line(str(self.dict[parameter]))

        return self.str

    def add(self, parameter):
        """
        Add a parameter or a list of parameters to the class.

        Parameters
        ----------
        parameter : FitParameter or list of FitParameter
            A single FitParameter or a list of FitParameter objects to be added.
        Returns
        -------

        """
        if isinstance(parameter, list):
            for param in parameter:
                self.dict[param.name] = param
        else:
            self.dict[parameter.name] = parameter

    def remove(self, name):
        """
        Removes a parameter from the Object.

        Parameters
        ----------
        name : str
            Name of the parameter to be removed.

        Returns
        -------

        """

        self.dict.pop(name)

    def change_value(self, name, value):
        """
        Implementation to change the value of a parameter.

        Parameters
        ----------
        name : str
            Name of the parameter which value is to be changed.
        value : float
            The new value that is to be assigned to the parameter.

        Returns
        -------

        """

        self.dict[name].value = value

    def valuesdict(self):
        """
        Creates a dictionary containing all FitParameter objects, with the keys being the respective names.

        Returns
        -------
        dict
        """
        valuesdict = {}
        for parameter in self.dict:
            valuesdict[parameter] = self.dict[parameter].value

        return valuesdict


class Fit:
    """
    Fit works as an object oriented wrapper for scipy's least_squares() implementation and is heavily inspired by
    lmfit. You create all necessary parameters as FitParameter objects and put them into a single FitParameters object.
    The FitParameters object and the residual function are then passed to the VariFit object. This grants a high
    flexibility, as the user can fix any parameters to a set value, which is then excluded from the fitting procedure.

    Parameters
    ----------
    func : function
        Residual function provided by the user, which returns a vector of the individual point-wise residuals.
    parameters : FitParameters
        FitParameters object which contains all FitParameter objects.

    Example
    --------
    import numpy as np

    def x_parab(x, a, b, c):
    return a * x ** 2 + b * x + c

    def noisy(x, stdev):
        return x + np.random.normal(0, stdev, len(x))


    x_data = np.linspace(-10, 10, 200)
    y_data = noisy(x_parab(x_data, 1.2, -4, 10), 5)
    y_error = 10


    def x_residual(param):
        values = param.valuesdict()
        a = values['a']
        b = values['b']
        c = values['c']

        result = x_parab(x_data, a, b, c) - y_data
        return result


    parameters_fit = FitParameters()
    parameters_fit.add(FitParameter('a', 1))
    parameters_fit.add(FitParameter('b', -5, vary=True))
    parameters_fit.add(FitParameter('c', 8, vary=True))

    fitting = Fit(x_residual, parameters_fit)
    fitting.least_squares()

    print(fitting.result)
    """

    def __init__(self, func, parameters):
        self.func = func
        self.parameters = parameters
        self.result = FitResult()

    def least_squares(self, method='trf', verbose=0, precision=8, max_nfev=100):
        """
        Wrapper of scipy's least_squares function, which is based on MINPACK.

        Parameters
        ----------
        method : str
            Method which is passed to scipy's least_squares function: {‘trf’, ‘dogbox’, ‘lm’}, optional

        verbose : {0, 1, 2}
            Level of algorithm’s verbosity:
            0 (default) : work silently.
            1 : display a termination report.
            2 : display progress during iterations (not supported by ‘lm’ method).

        precision : int
            Floating Point precision (in digits). Used for fitting procedure and rounding in print-out.


        Returns
        -------
        FitResult
        """

        parameters_vary_name = []
        all_parameter_names = []
        initial_condition = []
        lower_bound = []
        upper_bound = []

        for parameter in self.parameters.dict:
            all_parameter_names.append(parameter)

            self.parameters[parameter].precision = precision

            self.result.add(self.parameters.dict[parameter], self.parameters.dict[parameter].value, 0)

            if self.parameters.dict[parameter].vary:
                parameters_vary_name.append(parameter)

                initial_condition.append(self.parameters.dict[parameter].value)
                lower_bound.append(self.parameters.dict[parameter].min)
                upper_bound.append(self.parameters.dict[parameter].max)

        bounds = (lower_bound, upper_bound)

        def residual_wrapper(parameter_fit):
            """
            Wrapper for the user supplied residual function.

            Parameters
            ----------
            parameter_fit: list of floats
                List of floats from scipy's least_squares function.

            Returns
            -------
            list of floats
            """
            for name, value in zip(parameters_vary_name, parameter_fit):
                self.parameters.change_value(name, value)

            return self.func(self.parameters)

        prec_exp = 10**(-precision)

        result_lq = least_squares_scipy(residual_wrapper, initial_condition, jac='2-point', bounds=bounds, method=method,
                                        ftol=prec_exp, xtol=prec_exp, gtol=prec_exp, verbose=verbose, max_nfev=max_nfev)

        # marked section below was copied from scipy curve_fit
        # ----------------------------
        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(result_lq.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(result_lq.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
        # ----------------------------

        stdev_error = np.sqrt(np.diag(pcov))
        popt = result_lq.x

        for name, value, error in zip(parameters_vary_name, popt, stdev_error):
            self.result.change_value(name, value)
            self.result.change_stderr(name, error)
            self.result.pcov_names.append(name)

        self.result.nfev = result_lq.nfev
        self.result.status = result_lq.status
        self.result.active_mask = result_lq.active_mask
        self.result.pcov = pcov
        self.result.chi_sqr = 2 * result_lq.cost  # least_squares in scipy works with half chi-squared loss

        return self.result


class FitResult:
    """
    FitResult provides a simple class that can be used to access all necessary parameters of the fitting procedure.
    """

    def __init__(self):
        self.nfev = None
        self.status = -2  # the return values for 'status' from scipy least_squares start at -1
        self.active_mask = None  # label for each fitted parameters, (0: no constraint, -1: lower bound, 1: upper bound)
        self.chi_sqr = None
        self.pcov = None
        self.pcov_names = []
        self.result_parameters = {}
        self.str = "\nFit Result:\n"
        """
        The precision is set to None by default, and is later set to the smallest precision of the parameters. The 
        precision is only used for the terminal print-out.
        """
        self.precision = None

    def __iter__(self):
        """
        Class iterator.

        Returns
        -------

        """
        return iter(self.result_parameters)

    def __getitem__(self, name):
        """
        Returns ResultParameter.

        Parameters
        ----------
        name : str
            Name of the ResultParameter

        Returns
        -------
        FitParameter
        """

        return self.result_parameters[name]

    def add(self, parameter: FitParameter, init, stderr):
        """
        Add a FitParameter object with the respective init and stderr values,
        this is than combined into a new ResultParameter object.

        Parameters
        ----------
        parameter : FitParameter
            FitParameter object to be added.
        init : float
            Initial starting value used in the fitting procedure.
        stderr : float
            stderr of the parameter (from covariance matrix).

        Returns
        -------

        """
        self.result_parameters[parameter.name] = (ResultParameter(parameter.name,
                                                                  parameter.value,
                                                                  parameter.vary,
                                                                  parameter.min,
                                                                  parameter.max,
                                                                  init,
                                                                  stderr,
                                                                  parameter.precision))

        """
        Set precision to the lowest precision from all parameters. 
        Usually all parameters will have the same precision after the fitting procedure.
        """
        if self.precision is None:
            self.precision = parameter.precision

        elif self.precision > parameter.precision:
                self.precision = parameter.precision

    def change_value(self, name, value):
        """
        Implementation to change the value of a parameter.

        Parameters
        ----------
        name : str
            Name of the parameter which value is to be changed.
        value : float
            The new value that is to be assigned to the parameter.

        Returns
        -------

        """
        self.result_parameters[name].value = value

    def change_stderr(self, name, stderr):
        """
        Implementation to change the stderr of a parameter.

        Parameters
        ----------
        name : str
            Name of the parameter which value is to be changed.
        stderr :
            The new stderr that is to be assigned to the parameter.

        Returns
        -------

        """
        self.result_parameters[name].stderr = stderr

    def __str__(self):
        """
        Default string implementation, which returns a structured string.

        Returns
        -------
        string
        """

        def add_line(line):
            """
            Add a line to the final string with a line break at the end.
            Parameters
            ----------
            line : str
                Line to be added

            Returns
            -------

            """
            self.str += line + "\n"

        add_line("-" * len(self.str))
        add_line(f"Evaluations: {self.nfev}")
        add_line(f"Chi-Squared: {round(self.chi_sqr,3)}")
        add_line("")
        add_line("Parameters")
        for parameter in self.result_parameters:
            add_line(str(self.result_parameters[parameter]))

        add_line("")
        add_line("Covariance Matrix:")
        add_line(f"{self.pcov_names}")
        with np.printoptions(precision=self.precision, suppress=True, sign="+", linewidth=100):
            add_line(str(self.pcov))

        return self.str
