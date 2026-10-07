"""Headless smoke test: every dashboard page renders without exceptions."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PAGES = ["Overview", "Universe", "Risk", "Trends", "Correlation", "Data & Methodology"]


@pytest.fixture()
def app(pipeline_results, monkeypatch):
    paths, _ = pipeline_results
    monkeypatch.setenv("STOCK_ANALYSIS_DATA_DIR", str(paths.processed))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("ticker", ["RELIANCE", "JIOFIN"])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page, ticker):
    app.sidebar.radio[0].set_value(page).run()
    app.sidebar.selectbox[0].set_value(ticker).run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]


def test_sector_filter_narrows_stock_list(app):
    app.sidebar.multiselect[0].set_value(["Information Technology"]).run()
    assert not app.exception
    assert set(app.sidebar.selectbox[0].options) == {"HCLTECH · HCL Technologies", "INFY · Infosys", "TCS · Tata Consultancy Services", "TECHM · Tech Mahindra"}


def test_missing_data_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_ANALYSIS_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert at.error and "Missing processed files" in at.error[0].value
