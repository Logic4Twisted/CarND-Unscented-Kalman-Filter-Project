#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  time_us_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.57;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  weights_ = VectorXd(2*n_aug_+1);
  weights_[0] = lambda_/(lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_[i] = 0.5/(lambda_ + n_aug_);
  }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //std::cout << "Process measurement " << meas_package.sensor_type_ << std::endl;
  //std::cout << "Raw data :" << meas_package.raw_measurements_ << std::endl;
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) return;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) return;
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = meas_package.raw_measurements_[0];
      float theta = meas_package.raw_measurements_[1];
      float ro_dot = meas_package.raw_measurements_[2];
      float px = ro*sin(theta);
      float py = ro*cos(theta);
      float v = ro_dot;
      float fi = theta;
      float fi_dot = 0;
      x_ << px, py, v, fi, fi_dot;
      P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    //std::cout << "x_ :" << x_ << std::endl;
    //std::cout << "P_ :" << P_ << std::endl;
    return;
  }
  double delta_t = 1.0e-6*(meas_package.timestamp_ - time_us_);
  Prediction(delta_t);
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  time_us_ = meas_package.timestamp_;
  //std::cout << "x_ :" << std::endl;
  //std::cout << x_ << std::endl;
  //std::cout << "P_ :" << std::endl;
  //std::cout << P_ << std::endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  //std::cout << "---------------------------- Prediction step "  << delta_t << std::endl;

  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0);
  x_aug.head(n_x_) = x_;
  //std::cout << "x_aug :" << std::endl;
  //std::cout << x_aug << std::endl;

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug.col(5)[5] = std_a_*std_a_;
  P_aug.col(6)[6] = std_yawdd_*std_yawdd_;
  //std::cout << "P_aug :" << std::endl;
  //std::cout << P_aug << std::endl;

  MatrixXd A = P_aug.llt().matrixL();
  A = A*sqrt(lambda_ + n_aug_);
  //std::cout << "A :" << std::endl;
  //std::cout << A << std::endl;

  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i+1)= x_aug + A.col(i);
    Xsig_aug.col(i+1+n_aug_)= x_aug - A.col(i);
  }
  //std::cout << "Xsig_aug :" << std::endl;
  //std::cout << Xsig_aug << std::endl;

  // predict sigma points
  double delta_t_sqr = delta_t * delta_t;
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Xsig_pred_.col(i) = Xsig_aug.col(i).head(5);
    float px = Xsig_aug.col(i)[0];
    float py = Xsig_aug.col(i)[1];
    float v = Xsig_aug.col(i)[2];
    float fi = Xsig_aug.col(i)[3];
    float fi_dot = Xsig_aug.col(i)[4];
    float v_a = Xsig_aug.col(i)[5];
    float fi_a = Xsig_aug.col(i)[6];
    if (fabs(fi_dot) < EPSILON) {
      Xsig_pred_.col(i)[0] += v * cos(fi) * delta_t + 0.5 * delta_t_sqr * cos(fi) * v_a;
      Xsig_pred_.col(i)[1] += v * sin(fi) * delta_t + 0.5 * delta_t_sqr * sin(fi) * v_a;
      Xsig_pred_.col(i)[2] += delta_t * v_a;
      Xsig_pred_.col(i)[3] += AngleNormalization(fi_dot * delta_t + 0.5 * delta_t_sqr * fi_a);
      Xsig_pred_.col(i)[4] += delta_t * fi_a;
    } else {
      Xsig_pred_.col(i)[0] += v / fi_dot * (sin(fi + fi_dot * delta_t) - sin(fi))
          + 0.5 * delta_t_sqr * cos(fi) * v_a;
      Xsig_pred_.col(i)[1] += v / fi_dot * (-cos(fi + fi_dot * delta_t) + cos(fi))
          + 0.5 * delta_t_sqr * sin(fi) * v_a;
      Xsig_pred_.col(i)[2] += delta_t * v_a;
      Xsig_pred_.col(i)[3] += AngleNormalization(fi_dot * delta_t + 0.5 * delta_t_sqr * fi_a);
      Xsig_pred_.col(i)[4] += delta_t * fi_a;
    }
  }
  //std::cout << "Xsig_pred_ :" << std::endl;
  //std::cout << Xsig_pred_ << std::endl;

  // predict state mean
  x_ = Xsig_pred_*weights_;

  // predict state covariance matrix
  P_.fill(0.0);
  A = MatrixXd(n_x_, 2 * n_aug_ + 1);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    A.col(i) = Xsig_pred_.col(i) - x_;
  }
  MatrixXd A_t = A.transpose();
  for (int i = 0; i < 2*n_aug_+1; i++) {
    A.col(i) *= weights_[i];
  }
  P_ = A*A_t;

  //std::cout << "x_ :" << std::endl;
  //std::cout << x_ << std::endl;
  //std::cout << "P_:" << std::endl;
  //std::cout << P_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //std::cout << "----------------------------- Update lidar" << std::endl;
  MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Zsig.col(i) = Xsig_pred_.col(i).head(2);
  }
  //std::cout << "Zsig :" << std::endl;
  //std::cout << Zsig << std::endl;

  VectorXd z_pred = VectorXd(2);
  z_pred = Zsig*weights_;
  //std::cout << "z_pred :" << std::endl;
  //std::cout << z_pred << std::endl;

  // measurement vector
  VectorXd z = VectorXd(2);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  //std::cout << "z :" << std::endl;
  //std::cout << z << std::endl;

  // calculate measurement covariance matrix S
  MatrixXd S = MatrixXd(2, 2);
  MatrixXd A = MatrixXd(2, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    A.col(i) = Zsig.col(i) - z_pred;
  }
  MatrixXd A_t = A.transpose();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    A.col(i) *= weights_[i];
  }
  MatrixXd R = MatrixXd(2, 2);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S = A * A_t + R;
  //std::cout << "S :" << std::endl;
  //std::cout << S << std::endl;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, 2);
  MatrixXd X_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    X_.col(i) = Xsig_pred_.col(i) - x_;
    X_.col(i) *= weights_[i];
  }
  MatrixXd Z_ = MatrixXd(2, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Z_.col(i) = Zsig.col(i) - z_pred;
  }
  MatrixXd Z_t = Z_.transpose();
  Tc = X_ * Z_t;
  //std::cout << "Tc :" << std::endl;
  //std::cout << Tc << std::endl;

  //calculate Kalman gain K;
  MatrixXd S_i = S.inverse();
  MatrixXd K = Tc*S_i;

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  x_ += K*z_diff;
  MatrixXd K_t = K.transpose();
  P_ -= K*S*K_t;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //transform sigma points into measurement space
  //std::cout << "----------------------------- Update radar" << std::endl;
  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    float px = Xsig_pred_.col(i)[0];
    float py = Xsig_pred_.col(i)[1];
    float v = Xsig_pred_.col(i)[2];
    float fi = Xsig_pred_.col(i)[3];
    float fi_dot = Xsig_pred_.col(i)[4];

    float ro = sqrt(px * px + py * py);
    float s;
    if (fabs(px) < EPSILON) {
      if (py > 0.0) {
        s = M_PI / 2;
      } else {
        s = -M_PI / 2;
      }
    } else
      s = atan2(py, px);
    s = AngleNormalization(s);
    float ro_dot;
    if (ro < EPSILON) {
      ro_dot = v;
    } else {
      ro_dot = (px * cos(fi) + py * sin(fi)) * v / ro;
    }
    Zsig.col(i) << ro, s, ro_dot;
  }
  //std::cout << "Zsig :" << std::endl;
  //std::cout << Zsig << std::endl;

  // calculate mean predicted measurement
  VectorXd z_pred = VectorXd(3);
  z_pred = Zsig*weights_;
  z_pred[1] = AngleNormalization(z_pred[1]);
  //std::cout << "z_pred :" << std::endl;
  //std::cout << z_pred << std::endl;

  // measurement vector
  VectorXd z = VectorXd(3);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];
  //std::cout << "z :" << std::endl;
  //std::cout << z << std::endl;


  // calculate measurement covariance matrix S
  MatrixXd S = MatrixXd(3, 3);
  MatrixXd A = MatrixXd(3, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    A.col(i) = Zsig.col(i) - z_pred;
  }
  MatrixXd A_t = A.transpose();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    A.col(i) *= weights_[i];
  }
  MatrixXd R = MatrixXd(3, 3);
  R << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;
  S = A * A_t + R;
  //std::cout << "S :" << std::endl;
  //std::cout << S << std::endl;


  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, 3);
  MatrixXd X_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    X_.col(i) = Xsig_pred_.col(i) - x_;
    X_.col(i) *= weights_[i];
  }
  MatrixXd Z_ = MatrixXd(3, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Z_.col(i) = Zsig.col(i) - z_pred;
  }
  MatrixXd Z_t = Z_.transpose();
  Tc = X_ * Z_t;
  //std::cout << "Tc :" << std::endl;
  //std::cout << Tc << std::endl;

  //calculate Kalman gain K;
  MatrixXd S_i = S.inverse();
  MatrixXd K = Tc*S_i;

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  z_diff[1] = AngleNormalization(z_diff[1]);
  x_ += K*z_diff;
  MatrixXd K_t = K.transpose();
  P_ -= K*S*K_t;
}

double UKF::AngleNormalization(double angle) {
  // if angle is to large then error is significant therefore just return 0
  if (fabs(angle) > 1.0/EPSILON) return 0.0;
  while (angle >  M_PI) angle -= M_PI;
  while (angle < -M_PI) angle += M_PI;
  return angle;
}
