/* asin example */
#include <stdio.h> /* printf */
#include <math.h>  /* asin */
#include <iostream>
using namespace std;
#define PI 3.14159265

extern "C" double c_compute_alpha(double theta, double gamma)
{
    double alpha;
    alpha = theta - gamma;
    return alpha;
}

extern "C" double c_compute_gamma(double vz, double norm_vz)
{
    double gamma;
    if (norm_vz > 0)
    {
        gamma = asin(vz / norm_vz);
    }
    else
    {
        gamma = 0;
    }
    return gamma;
}

extern "C" double c_compute_cx(double alpha, double mach, double cx_min, double mach_critic)
{
    const double halfC = 180 / M_PI;
    double cx;
    alpha = (alpha * halfC) + 5;
    cx = pow(alpha * 0.02, 2) + cx_min;
    if (mach < mach_critic)
    {
        return cx / (sqrt(1 - pow(mach, 2)));
    }
    if (mach < 1)
    {
        return cx * 15 * (mach - mach_critic) + (cx / (sqrt(1 - pow(mach, 2))));
    }
    else
    {
        cout << "Supersonic";
        return 0;
    }
}

int main()
{
    float cx;
    cx = c_compute_cx(0.1, 0.8, 0.095, 0.73);
    cout << cx;
}
