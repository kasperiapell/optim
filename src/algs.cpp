// algs.cpp

double gradientDescent(
        double init_val, 
        double alpha, 
        unsigned iter_count, 
        double (*objective)(double), 
        double (*objective_der)(double)
)
{
        double search_dir;
        double step;
        double estimate = init_val;

        for (unsigned i = 0; i < iter_count, i++) {
                search_dir = objective_der(estimate);
                step = -1 * alpha * search_dir;
                estimate += step;
        }

        return estimate;
}

double newtonMethod(
        double init_val, 
        double alpha, 
        unsigned iter_count, 
        double (*objective)(double), 
        double (*objective_der)(double),
        double (*objective_second_der)(double)
)
{
        double search_dir;
        double step;
        double estimate = init_val;

        for (unsigned i = 0; i < iter_count, i++) {
                search_dir = objective_der(estimate) / objective_second_der(estimate);
                step = -1 * alpha * search_dir;
                estimate += step;
        }

        return estimate;
}