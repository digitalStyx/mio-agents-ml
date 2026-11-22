"""Przygotowanie obrazu i odrzucanie plików spoza formatu (plan, część 3)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import DATA, read_csv, run_module
from digits.data import ImageFormatError, load_image


@pytest.fixture(scope="module")
def good_image():
    return DATA / read_csv(DATA / "train.csv")[0]["image_path"]


def test_good_image_gives_64_values_on_the_0_16_grid(good_image):
    x = load_image(good_image)
    assert x.shape == (64,)
    assert x.min() >= 0 and x.max() <= 1
    # README: piksel = round(v * 255 / 16) dla v z 0..16, a cecha = piksel / 255.
    allowed = {round(v * 255 / 16) for v in range(17)}
    assert set(np.round(x * 255).astype(int)) <= allowed
    # Kolejność wiersz po wierszu, bez żadnej transformacji.
    raw = np.asarray(Image.open(good_image)).reshape(64)
    assert np.array_equal(np.round(x * 255).astype(int), raw)


def _bad_files(tmp: "Path") -> dict:
    files = {}
    Image.new("L", (16, 16), 128).save(tmp / "size16.png")
    files["inny rozmiar 16x16"] = tmp / "size16.png"
    Image.new("L", (8, 7)).save(tmp / "size8x7.png")
    files["inny rozmiar 8x7"] = tmp / "size8x7.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(tmp / "rgb.png")
    files["kolor RGB"] = tmp / "rgb.png"
    Image.new("LA", (8, 8)).save(tmp / "la.png")
    files["szarość z alfą"] = tmp / "la.png"
    Image.new("P", (8, 8)).save(tmp / "palette.png")
    files["paleta"] = tmp / "palette.png"
    Image.new("I;16", (8, 8)).save(tmp / "gray16.png")
    files["16-bitowa szarość"] = tmp / "gray16.png"
    Image.new("L", (8, 8)).save(tmp / "jpeg.png", format="JPEG")
    files["JPEG z rozszerzeniem .png"] = tmp / "jpeg.png"
    Image.new("L", (8, 8)).save(tmp / "gray.bmp")
    files["BMP"] = tmp / "gray.bmp"
    (tmp / "text.png").write_text("to nie jest obraz\n")
    files["plik tekstowy"] = tmp / "text.png"
    (tmp / "empty.png").write_bytes(b"")
    files["pusty plik"] = tmp / "empty.png"
    files["brak pliku"] = tmp / "missing.png"
    return files


@pytest.fixture(scope="module")
def bad_files(tmp_path_factory):
    return _bad_files(tmp_path_factory.mktemp("bad"))


CASES = ["inny rozmiar 16x16", "inny rozmiar 8x7", "kolor RGB", "szarość z alfą", "paleta",
         "16-bitowa szarość", "JPEG z rozszerzeniem .png", "BMP", "plik tekstowy", "pusty plik", "brak pliku"]


@pytest.mark.parametrize("case", CASES)
def test_bad_file_raises(bad_files, case):
    with pytest.raises(ImageFormatError):
        load_image(bad_files[case])


@pytest.mark.parametrize("case", CASES)
def test_bad_file_cli_errors_without_digit(bad_files, case, tmp_path):
    # Model nie jest potrzebny do odrzucenia pliku, więc wskazujemy nieistniejący:
    # błąd ma dotyczyć formatu obrazu, a nie modelu.
    r = run_module("digits.predict", str(bad_files[case]), "--model", str(tmp_path / "none.joblib"))
    assert r.returncode == 2, r.stderr
    assert r.stdout.strip() == ""
    assert "błąd" in r.stderr
