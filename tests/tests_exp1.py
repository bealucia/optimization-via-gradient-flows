import unittest
import numpy as np
from src.problems.quadratic import quadratic_matrix, exact_flow
from src.methods.euler import forward_euler, backward_euler


class TestExperiment1(unittest.TestCase):
    def setUp(self):
        self.A = quadratic_matrix(100)
        self.x0 = np.array([1.0, -0.5])
        self.grad = lambda x: self.A @ x
        self.hess = lambda x: self.A

    def test_spectrum(self):
        np.testing.assert_allclose(np.linalg.eigvalsh(self.A), [1, 100], atol=1e-12)

    def test_forward_exact_discrete(self):
        h, n = 0.015, 20
        X, info = forward_euler(self.grad, self.x0, h, n)
        expected = np.linalg.matrix_power(np.eye(2) - h * self.A, n) @ self.x0
        np.testing.assert_allclose(X[-1], expected, atol=1e-12)
        self.assertEqual(info["iterations"], n)

    def test_backward_exact_discrete(self):
        h, n = 0.021, 20
        X, info = backward_euler(self.grad, self.hess, self.x0, h, n)
        P = np.linalg.inv(np.eye(2) + h * self.A)
        expected = np.linalg.matrix_power(P, n) @ self.x0
        np.testing.assert_allclose(X[-1], expected, atol=1e-11)
        self.assertLess(info["max_implicit_residual"], 1e-10)

    def test_exact_flow(self):
        X = exact_flow(self.A, self.x0, [0.0])
        np.testing.assert_allclose(X[0], self.x0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
