// algs.cpp
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

MatrixXd newtonMethod(
        MatrixXd init_val, 
        double alpha, 
        unsigned iter_count, 
        MatrixXd (*objective)(MatrixXd), 
        MatrixXd (*objective_der)(MatrixXd),
        MatrixXd (*objective_second_der)(MatrixXd)
)
{
        MatrixXd search_dir;
        MatrixXd step;
        MatrixXd estimate = init_val;
        MatrixXd A;

        for (unsigned i = 0; i < iter_count; i++) {
                A = objective_second_der(estimate);
                Eigen::FullPivLU<Eigen::MatrixXd> LU_decomp(A);
                // Revert to gradient descent if Hessian not invertible
                if (LU_decomp.isInvertible()) {
                        search_dir = A * objective_der(estimate);
                } else {
                        search_dir = objective_der(estimate);
                }
                step = -1 * alpha * search_dir;
                estimate += step;
        }

        return estimate;
}