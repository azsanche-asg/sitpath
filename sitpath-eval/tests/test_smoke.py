def test_package_imports():
    import sitpath_eval

    assert hasattr(sitpath_eval, "__version__")
