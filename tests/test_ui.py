"""End-to-end UI tests with Playwright against a running server.

    .venv/bin/python -m imac_rag.cli serve &      # in another terminal
    .venv/bin/python -m pytest tests/test_ui.py -v

Each answered question costs one Gemini generation request (free tier: ~20/day per model),
so the suite asks only three questions in total.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect, sync_playwright

BASE = os.getenv("RAG_UI_URL", "http://127.0.0.1:8000")
SHOTS = Path(__file__).parent / "screenshots"
ANSWER_TIMEOUT = 120_000


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture(scope="module")
def page(browser):
    ctx = browser.new_context(viewport={"width": 1366, "height": 900})
    pg = ctx.new_page()
    pg.console_errors = []
    pg.on("console", lambda m: m.type == "error" and pg.console_errors.append(m.text))
    pg.on("pageerror", lambda e: pg.console_errors.append(str(e)))
    pg.goto(BASE)
    yield pg
    ctx.close()


def last_bot(page: Page):
    return page.locator(".msg.bot").last


def wait_answer(page: Page, n_bots: int):
    expect(page.locator(".msg.bot")).to_have_count(n_bots)
    expect(last_bot(page).locator(".typing")).to_have_count(0, timeout=ANSWER_TIMEOUT)


def test_landing(page: Page):
    expect(page).to_have_title("iMac Report RAG")
    expect(page.locator("header h1")).to_have_text("iMac Environmental Report")
    expect(page.locator(".chip")).to_have_count(6)
    expect(page.locator("#sources .empty")).to_be_visible()
    SHOTS.mkdir(exist_ok=True)
    page.screenshot(path=SHOTS / "01_landing.png")


def test_example_chip_answers_table_question(page: Page):
    page.get_by_role("button", name="What is the iMac's power draw in sleep and idle at 115V?").click()
    expect(page.locator(".msg.user")).to_have_count(1)
    expect(page.locator(".examples")).to_have_count(0)  # welcome panel removed
    wait_answer(page, 1)
    bubble = last_bot(page).locator(".bubble")
    expect(bubble).not_to_contain_text("Error")
    expect(bubble).to_contain_text("0.57")   # sleep @115V
    expect(bubble).to_contain_text("20.6")   # idle display on @115V
    expect(bubble.locator(".cite").first).to_be_visible()
    expect(bubble.locator(".meta")).to_contain_text("s")
    page.screenshot(path=SHOTS / "02_table_answer.png")


def test_citation_opens_source_with_page_image(page: Page):
    cite = last_bot(page).locator(".cite").first
    n = cite.get_attribute("data-n")
    cite.click()
    src = page.locator(".src.active")
    expect(src).to_have_count(1)
    expect(src.locator(".head")).to_contain_text(f"[{n}]")
    expect(src.locator(".badge")).to_be_visible()
    img = src.locator("img")
    expect(img).to_be_visible()
    page.wait_for_function("img => img.complete && img.naturalWidth > 0", arg=img.element_handle())
    page.screenshot(path=SHOTS / "03_citation_open.png")


def test_follow_up_uses_history(page: Page):
    page.locator("#q").fill("And at 230V?")
    page.keyboard.press("Enter")
    wait_answer(page, 2)
    bubble = last_bot(page).locator(".bubble")
    expect(bubble).not_to_contain_text("Error")
    expect(bubble).to_contain_text("0.59")   # sleep @230V
    expect(bubble).to_contain_text("20.9")   # idle @230V


def test_footnote_question_and_send_button(page: Page):
    page.locator("#q").fill("Is the iMac's AC power cord PVC-free in every country?")
    page.locator("#send").click()
    expect(page.locator("#send")).to_be_disabled()  # no double submit while waiting
    wait_answer(page, 3)
    bubble = last_bot(page).locator(".bubble")
    for country in ("India", "Thailand", "Korea"):
        expect(bubble).to_contain_text(country)
    expect(page.locator("#send")).to_be_enabled()
    expect(page.locator("#sources .badge", has_text="footnote").first).to_be_visible()
    page.screenshot(path=SHOTS / "04_footnote_answer.png", full_page=True)


def test_no_console_errors(page: Page):
    assert page.console_errors == [], page.console_errors


def test_mobile_layout_has_no_horizontal_scroll(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    pg = ctx.new_page()
    pg.goto(BASE)
    expect(pg.locator(".chip").first).to_be_visible()
    overflow = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow <= 0, f"horizontal overflow of {overflow}px"
    pg.screenshot(path=SHOTS / "05_mobile.png", full_page=True)
    ctx.close()


def test_dark_mode_renders(browser):
    ctx = browser.new_context(color_scheme="dark")
    pg = ctx.new_page()
    pg.goto(BASE)
    bg = pg.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == "rgb(20, 20, 20)", bg
    pg.screenshot(path=SHOTS / "06_dark.png")
    ctx.close()
