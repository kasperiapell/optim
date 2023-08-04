// algs.cpp

double gradientDescent(
        double init_val, 
        double alpha, 
        uint iter_count, 
        double (*objective)(double), 
        double (*objective_der)(double)
)
{
        double search_dir;
        double step;
        double estimate = init_val;

        for (uint i = 0; i < iter_count, i++) {
                search_dir = objective_der(estimate);
                step = -1 * alpha * search_dir;
                estimate += step;
        }

        return estimate;
}