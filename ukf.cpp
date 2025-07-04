#include "ukf.h"

UKF::UKF()
    : is_initialized_(false),
      n_x_(5),
      n_aug_(7),
      lambda_(3 - n_aug_),
      x_(Eigen::VectorXd::Zero(5)),
      P_(Eigen::MatrixXd::Identity(5,5)),
      weights_(Eigen::VectorXd(2 * n_aug_ + 1)),
      Xsig_pred_(Eigen::MatrixXd(n_x_, 2 * n_aug_ + 1)),
      std_a_(1.0),
      std_yawdd_(0.5) {
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
        weights_(i) = 0.5 / (n_aug_ + lambda_);
    }
}

void UKF::Initialize(const Eigen::VectorXd &x, const Eigen::MatrixXd &P) {
    x_ = x;
    P_ = P;
    is_initialized_ = true;
}

void UKF::GenerateAugmentedSigmaPoints(Eigen::MatrixXd &Xsig_aug) {
    Eigen::VectorXd x_aug = Eigen::VectorXd::Zero(n_aug_);
    x_aug.head(n_x_) = x_;

    Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

    Eigen::MatrixXd L = P_aug.llt().matrixL();

    Xsig_aug.col(0) = x_aug;
    double factor = std::sqrt(lambda_ + n_aug_);
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1) = x_aug + factor * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - factor * L.col(i);
    }
}

void UKF::SigmaPointPrediction(const Eigen::MatrixXd &Xsig_aug, double delta_t) {
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        double px_p, py_p;
        if (std::fabs(yawd) > 1e-3) {
            px_p = p_x + v / yawd * (std::sin(yaw + yawd * delta_t) - std::sin(yaw));
            py_p = p_y + v / yawd * (-std::cos(yaw + yawd * delta_t) + std::cos(yaw));
        } else {
            px_p = p_x + v * delta_t * std::cos(yaw);
            py_p = p_y + v * delta_t * std::sin(yaw);
        }

        double v_p = v + nu_a * delta_t;
        double yaw_p = yaw + yawd * delta_t + 0.5 * nu_yawdd * delta_t * delta_t;
        double yawd_p = yawd + nu_yawdd * delta_t;

        px_p += 0.5 * nu_a * delta_t * delta_t * std::cos(yaw);
        py_p += 0.5 * nu_a * delta_t * delta_t * std::sin(yaw);

        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
}

void UKF::NormalizeAngle(double &angle) {
    while (angle > M_PI) angle -= 2. * M_PI;
    while (angle < -M_PI) angle += 2. * M_PI;
}

void UKF::PredictMeanAndCovariance() {
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        NormalizeAngle(x_diff(3));
        P_ += weights_(i) * x_diff * x_diff.transpose();
    }
}

void UKF::Predict(double delta_t) {
    Eigen::MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
    GenerateAugmentedSigmaPoints(Xsig_aug);
    SigmaPointPrediction(Xsig_aug, delta_t);
    PredictMeanAndCovariance();
}

void UKF::UpdateLidar(const Eigen::VectorXd &z, const Eigen::MatrixXd &R) {
    const int n_z = 2;
    Eigen::MatrixXd Zsig = Xsig_pred_.topRows(n_z);

    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(n_z);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    Eigen::MatrixXd S = R;
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        S += weights_(i) * z_diff * z_diff.transpose();
    }

    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        NormalizeAngle(x_diff(3));
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    Eigen::MatrixXd K = Tc * S.inverse();
    Eigen::VectorXd z_diff = z - z_pred;
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
}

