#ifndef UKF_H
#define UKF_H
#include <Eigen/Dense>

class UKF {
public:
    UKF();
    void Initialize(const Eigen::VectorXd &x, const Eigen::MatrixXd &P);
    void Predict(double delta_t);
    void UpdateLidar(const Eigen::VectorXd &z, const Eigen::MatrixXd &R);
    const Eigen::VectorXd &state() const { return x_; }
    const Eigen::MatrixXd &covariance() const { return P_; }

private:
    void GenerateAugmentedSigmaPoints(Eigen::MatrixXd &Xsig_aug);
    void SigmaPointPrediction(const Eigen::MatrixXd &Xsig_aug, double delta_t);
    void PredictMeanAndCovariance();
    void NormalizeAngle(double &angle);

    bool is_initialized_;
    int n_x_;
    int n_aug_;
    double lambda_;

    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::VectorXd weights_;
    Eigen::MatrixXd Xsig_pred_;

    double std_a_;
    double std_yawdd_;
};

#endif // UKF_H
