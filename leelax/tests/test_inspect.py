def test_env_inspect_importable():
    # smoke test: we just want to ensure the CLI module exists and imports
    import leelax.env.inspect  # noqa: F401
