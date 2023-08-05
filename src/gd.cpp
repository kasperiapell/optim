// gd.cpp
#include <Eigen/Dense>

using Eigen::MatrixXd;

MatrixXd gradientDescent(
        MatrixXd init_val, 
        double alpha, 
        unsigned iter_count, 
        MatrixXd (*objective)(MatrixXd), 
        MatrixXd (*objective_der)(MatrixXd)
)
{
        MatrixXd search_dir;
        MatrixXd step;
        MatrixXd estimate = init_val;

        for (unsigned i = 0; i < iter_count; i++) {
                search_dir = objective_der(estimate);
                step = -1 * alpha * search_dir;
                estimate += step;
        }

        return estimate;
}