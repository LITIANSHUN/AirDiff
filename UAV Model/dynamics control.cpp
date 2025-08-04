/*******************************************************
 *  6-DOF Package-Delivery Quadcopter – Minimal C++17
 *
 *  Author : your-name
 *  Compile: g++ -std=c++17 quadcopter_delivery.cpp -o quad
 *******************************************************/

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// -----------------------------------------------------
//  Constants & aliases
// -----------------------------------------------------
constexpr double kGravity      = 9.80665;   // m/s²
constexpr double kMassQuad     = 1.2;       // kg
constexpr double kMassPkg      = 0.5;       // kg
constexpr double kArmLength    = 0.225;     // m (motor ↔ CoG)
constexpr double kInertia[3]   = {0.01, 0.01, 0.02}; // Ixx, Iyy, Izz
constexpr double kThrustCoeff  = 1.858e-05; // N/(rpm)²
constexpr double kDragCoeff    = 1.140e-06; // N⋅m/(rpm)²
constexpr double kDt           = 1e-3;      // 1 ms sim step
constexpr double kMaxRPM       = 10000.0;

using Vec3   = std::array<double, 3>;
using Mat3x3 = std::array<std::array<double, 3>, 3>;

// -----------------------------------------------------
//  Utility
// -----------------------------------------------------
inline double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]};
}

inline Mat3x3 eulerToRot(const Vec3& euler) {  // XYZ order [rad]
    double roll  = euler[0];
    double pitch = euler[1];
    double yaw   = euler[2];

    double cr = cos(roll),  sr = sin(roll);
    double cp = cos(pitch), sp = sin(pitch);
    double cy = cos(yaw),   sy = sin(yaw);

    Mat3x3 R;
    R[0] = {cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr};
    R[1] = {sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr};
    R[2] = {-sp,    cp*sr,             cp*cr};
    return R;
}

inline Vec3 matVec(const Mat3x3& M, const Vec3& v) {
    return {M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]};
}

// -----------------------------------------------------
//  PID controller
// -----------------------------------------------------
class PID {
public:
    PID(double kp, double ki, double kd)
        : kp_(kp), ki_(ki), kd_(kd), integral_(0), prev_(0) {}

    double update(double err, double dt) {
        integral_ += err * dt;
        double derivative = (err - prev_) / dt;
        prev_ = err;
        return kp_ * err + ki_ * integral_ + kd_ * derivative;
    }
    void reset() { integral_ = 0; prev_ = 0; }

private:
    double kp_, ki_, kd_;
    double integral_, prev_;
};

// -----------------------------------------------------
//  Trajectory generator – 5th-order poly
// -----------------------------------------------------
struct Waypoint { Vec3 pos; double t_arrive; };

class TrajectoryGenerator {
public:
    explicit TrajectoryGenerator(std::vector<Waypoint> wps)
        : wps_(std::move(wps)), idx_(0), t0_(0) {
        if (wps_.empty()) throw std::runtime_error("no waypoints");
    }

    Vec3 pos(double t) {
        while (idx_ + 1 < wps_.size() && t > wps_[idx_+1].t_arrive) ++idx_;
        if (idx_ + 1 >= wps_.size()) return wps_.back().pos; // hover at last
        double T = wps_[idx_+1].t_arrive - wps_[idx_].t_arrive;
        double tau = (t - wps_[idx_].t_arrive) / T;
        tau = clamp(tau, 0.0, 1.0);

        // 5th order poly with zero vel/acc at start/end
        double a0=0, a1=0, a2=0, a3=10, a4=-15, a5=6;
        Vec3 p0 = wps_[idx_].pos;
        Vec3 p1 = wps_[idx_+1].pos;
        Vec3 delta = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
        double poly = a0 + a1*tau + a2*tau*tau + a3*tau*tau*tau
                    + a4*tau*tau*tau*tau + a5*tau*tau*tau*tau*tau;
        return {p0[0] + delta[0]*poly,
                p0[1] + delta[1]*poly,
                p0[2] + delta[2]*poly};
    }

private:
    std::vector<Waypoint> wps_;
    size_t idx_;
    double t0_;
};

// -----------------------------------------------------
//  Quadcopter 6-DOF dynamics
// -----------------------------------------------------
class Quadcopter6DOF {
public:
    struct State {
        Vec3  pos;        // inertial [m]
        Vec3  vel;        // inertial [m/s]
        Vec3  euler;      // roll, pitch, yaw [rad]
        Vec3  omega;      // body rates [rad/s]
    };

    Quadcopter6DOF() {
        state_.pos   = {0,0,0};
        state_.vel   = {0,0,0};
        state_.euler = {0,0,0};
        state_.omega = {0,0,0};
        has_pkg_ = true;
    }

    State state() const { return state_; }
    bool  hasPackage() const { return has_pkg_; }

    void setMotorRPM(const Vec3& rpm) { rpm_des_ = rpm; } // only for test

    void step(const std::array<double,4>& rpm_cmd, double dt) {
        // saturate
        std::array<double,4> rpm;
        for (int i=0;i<4;++i) rpm[i] = clamp(rpm_cmd[i], 0.0, kMaxRPM);

        // thrust & moments
        double thrust = 0;
        for (double r : rpm) thrust += kThrustCoeff * r * r;

        Vec3 moment{0,0,0};
        moment[0] = kArmLength * kThrustCoeff * (rpm[1]*rpm[1] - rpm[3]*rpm[3]);
        moment[1] = kArmLength * kThrustCoeff * (rpm[2]*rpm[2] - rpm[0]*rpm[0]);
        moment[2] = kDragCoeff * (rpm[0]*rpm[0] - rpm[1]*rpm[1] +
                                  rpm[2]*rpm[2] - rpm[3]*rpm[3]);

        // gravity in body frame
        Vec3 grav_body = matVec(eulerToRot(state_.euler), {0,0,kGravity});

        // total mass
        double m = kMassQuad + (has_pkg_ ? kMassPkg : 0);

        // accel
        Vec3 acc = {0,0, -thrust/m};
        acc = matVec(eulerToRot(state_.euler), acc);
        acc[0] += grav_body[0];
        acc[1] += grav_body[1];
        acc[2] += grav_body[2];

        // angular accel
        Vec3 alpha;
        alpha[0] = moment[0] / kInertia[0];
        alpha[1] = moment[1] / kInertia[1];
        alpha[2] = moment[2] / kInertia[2];

        // integrate
        for (int i=0;i<3;++i) {
            state_.vel[i]  += acc[i] * dt;
            state_.pos[i]  += state_.vel[i] * dt;
            state_.omega[i] += alpha[i] * dt;
            state_.euler[i] += state_.omega[i] * dt;
        }
    }

    void releasePackage() { has_pkg_ = false; }

private:
    State state_;
    bool has_pkg_;
    Vec3 rpm_des_;
};

// -----------------------------------------------------
//  Cascaded controller
// -----------------------------------------------------
class Controller {
public:
    Controller()
        : pid_x_(1.2, 0.0, 0.5),
          pid_y_(1.2, 0.0, 0.5),
          pid_z_(0.8, 0.0, 0.5),
          pid_roll_(8.0, 0.0, 2.0),
          pid_pitch_(8.0, 0.0, 2.0),
          pid_yaw_(3.0, 0.0, 0.5) {}

    std::array<double,4> update(const Quadcopter6DOF::State& st,
                                const Vec3& pos_des,
                                double yaw_des,
                                double dt) {
        // position loops -> desired angles
        double ex = pos_des[0] - st.pos[0];
        double ey = pos_des[1] - st.pos[1];
        double ez = pos_des[2] - st.pos[2];

        double ux = pid_x_.update(ex, dt);
        double uy = pid_y_.update(ey, dt);
        double uz = pid_z_.update(ez, dt);

        double thrust_total = (kMassQuad+kMassPkg) * kGravity + uz;

        // desired roll/pitch from backstepping
        double roll_des  = clamp(asin(-uy * sin(st.euler[2]) + ux * cos(st.euler[2])) /
                                 sqrt(ux*ux + uy*uy + uz*uz), -0.5, 0.5);
        double pitch_des = clamp(asin((ux * sin(st.euler[2]) + uy * cos(st.euler[2])) /
                                      sqrt(ux*ux + uy*uy + uz*uz)), -0.5, 0.5);

        // attitude loops
        double roll_err  = roll_des  - st.euler[0];
        double pitch_err = pitch_des - st.euler[1];
        double yaw_err   = yaw_des   - st.euler[2];

        double tau_roll  = pid_roll_.update(roll_err, dt);
        double tau_pitch = pid_pitch_.update(pitch_err, dt);
        double tau_yaw   = pid_yaw_.update(yaw_err, dt);

        // mixing
        double w_sq = thrust_total / (4 * kThrustCoeff);
        double w1_sq = w_sq - tau_pitch/(2*kArmLength*kThrustCoeff) - tau_roll/(2*kArmLength*kThrustCoeff) - tau_yaw/(4*kDragCoeff);
        double w2_sq = w_sq + tau_pitch/(2*kArmLength*kThrustCoeff) - tau_roll/(2*kArmLength*kThrustCoeff) + tau_yaw/(4*kDragCoeff);
        double w3_sq = w_sq + tau_pitch/(2*kArmLength*kThrustCoeff) + tau_roll/(2*kArmLength*kThrustCoeff) - tau_yaw/(4*kDragCoeff);
        double w4_sq = w_sq - tau_pitch/(2*kArmLength*kThrustCoeff) + tau_roll/(2*kArmLength*kThrustCoeff) + tau_yaw/(4*kDragCoeff);

        return {sqrt(std::max(w1_sq,0.0)),
                sqrt(std::max(w2_sq,0.0)),
                sqrt(std::max(w3_sq,0.0)),
                sqrt(std::max(w4_sq,0.0))};
    }

private:
    PID pid_x_, pid_y_, pid_z_;
    PID pid_roll_, pid_pitch_, pid_yaw_;
};

// -----------------------------------------------------
//  Main simulation loop
// -----------------------------------------------------
int main() {
    Quadcopter6DOF quad;
    Controller ctrl;

    std::vector<Waypoint> wps = {
        {{-5,-3,0}, 0},
        {{-2.5,-1.5,6}, 5},
        {{0,0,6}, 10},
        {{2.5,1.5,6}, 15},
        {{5,3,1}, 20}
    };
    TrajectoryGenerator traj(wps);

    double t = 0;
    const double t_final = 25.0;

    // simple console logger
    std::cout << "time,x,y,z,roll,pitch,yaw,hasPkg\n";
    std::cout << std::fixed << std::setprecision(3);

    while (t <= t_final) {
        Vec3 pos_des = traj.pos(t);
        std::array<double,4> rpm = ctrl.update(quad.state(), pos_des, 0.0, kDt);

        quad.step(rpm, kDt);

        // drop package near last waypoint
        if (t > 19.9 && quad.hasPackage()) {
            quad.releasePackage();
        }

        auto st = quad.state();
        std::cout << t << ',' << st.pos[0] << ',' << st.pos[1] << ',' << st.pos[2] << ','
                  << st.euler[0] << ',' << st.euler[1] << ',' << st.euler[2] << ','
                  << quad.hasPackage() << '\n';

        t += kDt;
    }

    std::cout << "Simulation finished.\n";
    return 0;
}