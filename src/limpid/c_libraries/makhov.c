#include <math.h>
#include <gsl/gsl_integration.h>

struct integrand_params {double energy; double rho; double a; double n; double m; double u; double thickness; };

double makhov_profile (double z, double energy, double rho, double a, double n, double m) {
    /*
    def makhov_profile(z, e, rho, a, n, m):
        z_avg = a / rho * e ** n * 10
        z_0 = z_avg / math.gamma(1 / m + 1)

        return (m * z ** (m - 1)) / (z_0 ** m) * np.exp(-(z / z_0) ** m)
    */

    double z_avg = a / rho * pow(energy, n) * 10;
    double z_0 = z_avg / tgamma(1 / m + 1);

    return (m * pow(z,(m - 1))) / pow(z_0, m) * exp(-pow((z / z_0), m));
}

double makhov_profile_integrated (double end, double offset, double energy, double rho, double a, double n, double m) {
    /*
     * integrates from 0 to the end
     */
    double z_avg = a / rho * pow(energy, n) * 10;
    double z_0 = z_avg / tgamma((1 / m) + 1);

    return exp(- pow(offset/z_0, m)) - exp(- pow((end + offset)/z_0, m));
}

double makhov_profile_depth(double implanted, double energy, double rho, double a, double n, double m) {
    /*
     def makhov_depth(rho, a, n, m, implanted, energy):
        z_avg = a / rho * energy ** n * 10
        z_0 = z_avg / math.gamma(1 / m + 1)

        # deals with infinite offsets, the error made by this implementation is negligible
        if implanted > 1 - 1E-15:
            implanted = 1 - 1E-15

        return z_0 * np.power(-np.log(1 - implanted), 1 / m)
    */

    double z_avg = a / rho * pow(energy, n) * 10;
    double z_0 = z_avg / tgamma(1 / m + 1);

    if (implanted > (double) 1.0 - (double) 1e-15) {
        implanted = (double) 1.0 - (double) 1e-15;
    }

    return z_0 * pow(-log(1 - implanted), 1 / m);
}

double concentration_left (double z, double u, double thickness) {
    /*
    def concentration_left(z, u, thickness):
        return (np.exp(-u * z) - np.exp(u * (z - 2 * thickness))) / (1 - np.exp(-2 * u * thickness))
    */
    return (exp(-u * z) - exp(u * (z - 2 * thickness))) / (- expm1(-2 * u * thickness));
}

double concentration_right (double z, double u, double thickness) {
    /*
    def concentration_right(z, u, thickness):
        return (np.exp(u * z) - np.exp(- u * z)) / (np.exp(u * thickness) - np.exp(- u * thickness))
    */

    if (u*thickness > 30) {
        return exp(u * (z - thickness));
    }
    if (u*z > 30) {
        return exp(u * (z - thickness))/(-expm1(- 2 * u * thickness));
    }

    return (exp(u * z) - exp(- u * z)) / (exp(u * thickness) - exp(- u * thickness));
}

double integrand_left (double z, void * p) {

    struct integrand_params * params = (struct integrand_params *)p;
    double energy = (params->energy);
    double rho = (params->rho);
    double a = (params->a);
    double n = (params->n);
    double m = (params->m);
    double u = (params->u);
    double thickness = (params->thickness);

    return makhov_profile(z, energy, rho, a, n, m) * concentration_left(z, u, thickness);
}

double integrand_right (double z, void * p) {

    struct integrand_params * params = (struct integrand_params *)p;
    double energy = (params->energy);
    double rho = (params->rho);
    double a = (params->a);
    double n = (params->n);
    double m = (params->m);
    double u = (params->u);
    double thickness = (params->thickness);

    return makhov_profile(z, energy, rho, a, n, m) * concentration_right(z, u, thickness);
}

double integral_left (double end, double offset, double thickness, double energy, double rho, double a,
                      double n, double m, double u, double eps) {

    gsl_integration_workspace * w_l = gsl_integration_workspace_alloc (10000);

    double result_l, error_l;
    struct integrand_params params = {energy, rho, a, n, m, u, thickness};

    gsl_function IntegrandLeft;
    IntegrandLeft.function = &integrand_left;
    IntegrandLeft.params = &params;

    if (isfinite(end)) {
        gsl_integration_qag (&IntegrandLeft, offset, end + offset, eps, eps, 10000,
                             6,w_l, &result_l, &error_l);
    } else {
        gsl_integration_qagiu (&IntegrandLeft, offset, eps, eps, 10000,w_l, &result_l, &error_l);
    }

    gsl_integration_workspace_free (w_l);

    return result_l;
}

double integral_right (double end, double offset, double thickness, double energy, double rho, double a,
                       double n, double m, double u, double eps) {

    gsl_integration_workspace * w_r = gsl_integration_workspace_alloc (10000);

    double result_r, error_r;
    struct integrand_params params = {energy, rho, a, n, m, u, thickness};

    gsl_function IntegrandRight;
    IntegrandRight.function = &integrand_right;
    IntegrandRight.params = &params;

    if (isfinite(end)) {
        gsl_integration_qag (&IntegrandRight, offset, end + offset, eps, eps, 10000,
                             6,w_r, &result_r, &error_r);
    } else {
        result_r = 0.0;
    }

    gsl_integration_workspace_free (w_r);

    return result_r;
}

int makhov_integration_func(int num_of_energies, double *c_left, double *c_right, double *c_ann, double *c_implanted, double *offsets,
                  double *prev_implanted, double *energies, double thickness, double rho, double a, double n, double m, double u,
                  double eps) {
    /*
        def integral(f, e):
            return integrate.quad(f, 0, thickness, args=(e,), epsabs=prec_exp, epsrel=prec_exp)[0]

        c_left = np.zeros_like(energies)
        c_right = np.zeros_like(energies)
        c_ann = np.zeros_like(energies)
        c_implanted = np.zeros_like(energies)
        offsets = np.zeros_like(energies)

        # translate loop to C:
        # offsets, c_left, c_right, c_implanted = mighty_c_func(prev_implanted, energies, thickness, u, params)
        for i, energy in enumerate(energies):
            offset = self.implantation_profile.get_depth(prev_implanted[i], energy)
            offsets[i] = offset
            c_left[i] = integral(lambda z, e: self.implantation_profile(z + offset, e) *
                                              self.concentration_left(z, u, thickness), energy)
            c_right[i] = integral(lambda z, e: self.implantation_profile(z + offset, e) *
                                               self.concentration_right(z, u, thickness), energy)
            c_implanted[i] = integral(lambda z, e: self.implantation_profile(z + offset, e), energy)
            c_ann[i] = c_implanted[i] - c_left[i] - c_right[i]
     */
    for (int i = 0; i < num_of_energies; ++i) {
        offsets[i] = makhov_profile_depth(prev_implanted[i], energies[i], rho, a, n, m);
        c_left[i] = integral_left(thickness, offsets[i], thickness, energies[i], rho, a, n, m, u, eps);
        c_right[i] = integral_right(thickness, offsets[i], thickness, energies[i], rho, a, n, m, u, eps);
        c_implanted[i] = makhov_profile_integrated(thickness, offsets[i], energies[i], rho, a, n, m);
        c_ann[i] = c_implanted[i] - c_left[i] - c_right[i];
    }

    return 0;
}

