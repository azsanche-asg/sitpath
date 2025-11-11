def test_package_imports():
    import sitpath_eval

    assert hasattr(sitpath_eval, "__version__")


def test_dependencies_available():
    import torch  # noqa: F401
    import numpy as np  # noqa: F401

    assert isinstance(torch.__version__, str)


def test_random_ade_example_returns_number():
    import numpy as np

    rng = np.random.default_rng(seed=0)
    preds = rng.random((8, 2))
    targets = rng.random((8, 2))
    distances = np.linalg.norm(preds - targets, axis=1)
    ade = float(np.mean(distances))

    assert ade > 0.0
