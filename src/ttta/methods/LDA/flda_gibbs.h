#include <math.h>

double digamma(double x) {
    double r = 0.0;

    while (x <= 5.0) {
        r -= 1.0 / x;
        x += 1.0;
    }

    double f = 1.0 / (x * x);
    double t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0 + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617.0 / 8160.0)))))));

    return r + log(x) - 0.5 / x + t;
}
