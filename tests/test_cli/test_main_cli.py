from importlib import import_module


def test_main_registers_commands():
    mod = import_module("openenv_cli.__main__")
    app = getattr(mod, "app")
    # Typer app stores registered commands in info.commands
    names = [c.name for c in app.registered_commands]
    # Ensure both commands are registered
    assert "push" in names
    assert "init" in names


