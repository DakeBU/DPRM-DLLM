import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "integrations/omni_diffusion/scripts/stage_omni_model_code.py"
)
SPEC = importlib.util.spec_from_file_location("stage_omni_model_code", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stage_model_links_weights_and_copies_generation_code(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "generation_utils.py").write_text("old = True\n", encoding="utf-8")
    overlay = tmp_path / "released_generation_utils.py"
    overlay.write_text("released = True\n", encoding="utf-8")

    output = tmp_path / "staged"
    manifest = MODULE.stage_model(source, output, overlay)

    assert manifest.is_file()
    assert (output / "model.safetensors").is_symlink()
    assert (output / "model.safetensors").read_bytes() == b"weights"
    assert not (output / "generation_utils.py").is_symlink()
    assert (output / "generation_utils.py").read_text() == "released = True\n"
