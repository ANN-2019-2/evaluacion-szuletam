from assigment import Regressor
import numpy as np
import pytest


def test_predict():
    reg = Regressor()
    reg.coef_ = np.array([1, 2, 3])
    reg.intercept_ = 2
    x = [[1, 2, 3]]
    assert reg.predict(x) == [16]


def test_initialize_weights():

    reg = Regressor(warm_start=False)
    reg.warm_start = False
    reg.coef_ = np.array([1, 2, 3])
    reg.intercept_ = 2
    reg.initialize_weights(n_features=3)
    assert reg.intercept_ == 0.0
    assert (reg.coef_ == np.zeros(3)).all()

    reg = Regressor(warm_start=True)
    reg.warm_start = True
    reg.coef_ = np.array([1, 2, 3])
    reg.intercept_ = 2.0
    reg.initialize_weights(n_features=3)
    assert reg.intercept_ == 2.0
    assert (reg.coef_ == np.array([1, 2, 3])).all()

    reg = Regressor(warm_start=True)
    reg.initialize_weights(n_features=3)
    assert reg.intercept_ == 0.0
    assert (reg.coef_ == np.zeros(3)).all()


@pytest.mark.parametrize(
    "loss, y_real, y_pred, epsilon, expected_result",
    [
        ("squared_loss", 2.0, 1.0, 0.001, 0.5),
        ("huber", 0.2, 0.1, 0.10, 0.0),
        ("huber", 0.2, 0.1, 0.01, 0.00095),
        ("epsilon_insensitive", 0.2, 0.1, 0.10, 0.0),
        ("epsilon_insensitive", 0.2, 0.1, 0.04, 0.06),
        ("squared_epsilon_insensitive", 0.2, 0.1, 0.10, 0.0),
        ("squared_epsilon_insensitive", 0.2, 0.1, 0.04, 0.0036),
    ],
)
def test_compute_loss(loss, y_real, y_pred, epsilon, expected_result):
    reg = Regressor(loss=loss, epsilon=epsilon)
    assert reg.compute_loss(y_real=y_real, y_pred=y_pred) == pytest.approx(
        expected_result
    )


@pytest.mark.parametrize(
    "loss, x, coef_, intercept_, y_real, epsilon, expected_gcoef, expected_gintercept",
    [
        (
            "squared_loss",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            -3.0,
            0.01,
            np.array([0.55, 1.1]),
            5.5,
        ),
        (
            "huber",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            -3.0,
            0.01,
            np.array([0.001, 0.002]),
            0.01,
        ),
        (
            "huber",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            0.1,
            0.01,
            np.array([-0.001, -0.002]),
            -0.01,
        ),
        (
            "epsilon_insensitive",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            -0.024,
            0.001,
            np.array([-0.1, -0.2]),
            -1.0,
        ),
        (
            "epsilon_insensitive",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            -0.25,
            0.001,
            0.0,
            0.0,
        ),
        (
            "epsilon_insensitive",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            -0.26,
            0.001,
            np.array([0.1, 0.2]),
            1.0,
        ),
        (
            "squared_epsilon_insensitive",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            -0.24,
            0.001,
            np.array([-0.0018, -0.0036]),
            -0.018,
        ),
        (
            "squared_epsilon_insensitive",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            -0.25,
            0.001,
            np.array([0.0, 0.0]),
            0.0,
        ),
        (
            "squared_epsilon_insensitive",
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            -0.3,
            -0.26,
            0.001,
            np.array([0.0018, 0.0036]),
            0.018,
        ),
    ],
)
def test_compute_loss_gradient(
    loss, x, coef_, intercept_, y_real, epsilon, expected_gcoef, expected_gintercept
):
    reg = Regressor(loss=loss, epsilon=epsilon)
    reg.coef_ = coef_
    reg.intercept_ = intercept_
    computed_gcoef, computed_gintercept = reg.compute_loss_gradient(x, y_real)
    assert expected_gcoef == pytest.approx(computed_gcoef)
    assert expected_gintercept == pytest.approx(computed_gintercept)


@pytest.mark.parametrize(
    "penalty, coef_, intercept_, l1_ratio, expected_result",
    [
        ("l2", np.array([0.1, 0.2]), -0.3, 0.1, np.array([0.1, 0.2])),
        ("l1", np.array([0.1, 0.2]), -0.3, 0.1, np.array([1, 1])),
        ("elasticnet", np.array([0.1, 0.2]), -0.3, 0.1, np.array([0.91, 0.92])),
    ],
)
def test_compute_penalty_gradient(
    penalty, coef_, intercept_, l1_ratio, expected_result
):
    reg = Regressor()
    reg.penalty = penalty
    reg.coef_ = coef_
    reg.intercept_ = intercept_
    reg.l1_ratio = l1_ratio
    assert pytest.approx(reg.compute_penalty_gradient()) == expected_result
