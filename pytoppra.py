import numpy as np
import cvxpy as cp
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Optional

class TOPPRA:
    def __init__(self, path: np.ndarray, vel_limits: np.ndarray, accel_limits: np.ndarray):
        self.path = path
        self.vel_limits = vel_limits
        self.accel_limits = accel_limits
        self.n_dof = path.shape[1]
        self.n_points = path.shape[0]
        
        # Initialize other necessary attributes
        self.s = np.linspace(0, 1, self.n_points)
        self.ds = 1.0 / (self.n_points - 1)
        self.path_derivative = self._compute_path_derivative()
        self.path_second_derivative = self._compute_path_second_derivative()

    def _compute_path_derivative(self) -> np.ndarray:
        return np.gradient(self.path, self.s, axis=0)

    def _compute_path_second_derivative(self) -> np.ndarray:
        return np.gradient(self._compute_path_derivative(), self.s, axis=0)

    def compute_parameterization(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_points
        x = cp.Variable(n)
        u = cp.Variable(n-1)
        
        constraints = self._compute_constraints(x, u)
        objective = cp.Minimize(cp.sum(1 / cp.sqrt(x)))
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            raise RuntimeError("Optimization problem did not converge.")
        
        s = self.s
        sd = np.sqrt(x.value)
        return s, sd

    def _compute_constraints(self, x: cp.Variable, u: cp.Variable) -> List[cp.constraints.Constraint]:
        constraints = []
        
        # Velocity limits
        for i in range(self.n_dof):
            constraints.append(cp.multiply(self.path_derivative[:, i]**2, x) <= self.vel_limits[i, 1]**2)
            constraints.append(cp.multiply(self.path_derivative[:, i]**2, x) >= self.vel_limits[i, 0]**2)
        
        # Acceleration limits
        for i in range(self.n_dof):
            constraints.append(
                cp.multiply(self.path_derivative[:, i], u) + 
                cp.multiply(self.path_second_derivative[:, i], x) <= self.accel_limits[i, 1]
            )
            constraints.append(
                cp.multiply(self.path_derivative[:, i], u) + 
                cp.multiply(self.path_second_derivative[:, i], x) >= self.accel_limits[i, 0]
            )
        
        # Continuity constraints
        constraints.append(x[1:] - x[:-1] - self.ds * u == 0)
        
        # Boundary conditions
        constraints.append(x[0] == 0)
        constraints.append(x[-1] == 0)
        
        return constraints

    def interpolate_trajectory(self, s: np.ndarray, sd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate the time-optimal trajectory.

        Args:
            s (np.ndarray): Path parameter.
            sd (np.ndarray): Path parameter derivative.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing time (t), position (q), velocity (qd), and acceleration (qdd).
        """
        t = np.cumsum(2 * np.diff(s) / (sd[:-1] + sd[1:]))
        t = np.insert(t, 0, 0)
        
        q = self.path
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)
        
        for i in range(self.n_dof):
            cs = CubicSpline(s, q[:, i])
            qd[:, i] = cs.derivative(1)(s) * sd
            qdd[:, i] = (cs.derivative(2)(s) * sd**2 + cs.derivative(1)(s) * np.gradient(sd, s))
        
        return t, q, qd, qdd


def main():
    # Example usage
    path = np.array([
        [0, 0],
        [1, 1],
        [2, 0],
        [3, 1],
        [4, 0]
    ])
    vel_limits = np.array([[-1, 1], [-1, 1]])
    accel_limits = np.array([[-2, 2], [-2, 2]])

    toppra = TOPPRA(path, vel_limits, accel_limits)
    s, sd = toppra.compute_parameterization()
    t, q, qd, qdd = toppra.interpolate_trajectory(s, sd)

    print("Time-optimal trajectory computed successfully.")
    print(f"Total time: {t[-1]:.2f} seconds")
    print(f"Number of points: {len(t)}")

if __name__ == "__main__":
    main()

