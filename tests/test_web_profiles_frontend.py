"""Browser regression checks for private chat restore and calendar scope labels."""
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

import pytest


playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profiles_browser():
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "web_demo")))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    with playwright.sync_playwright() as runtime:
        try:
            browser = runtime.chromium.launch(headless=True)
        except playwright.Error as error:
            server.shutdown()
            server.server_close()
            pytest.skip(f"Chromium is unavailable: {error}")
        try:
            yield browser, f"http://127.0.0.1:{server.server_address[1]}/static/index.html#perfiles"
        finally:
            browser.close()
            server.shutdown()
            server.server_close()
            worker.join()


def test_profiles_restore_private_chat_and_show_expired_session(profiles_browser):
    browser, url = profiles_browser
    page = browser.new_page(viewport={"width": 320, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    chats, calls = {}, []
    model = {"id": "sft-101", "label": "SFT", "method": "SFT-ramas", "seed": 101,
             "panel": "validation_selected", "epoch": 1, "eligible": True, "available": True}
    catalog = {
        "profiles_v5": True, "scenarios": [], "guard_scenarios": [], "chat_choices": [],
        "chat_models": [], "live_available": False,
        "shared_budget": {"cases": [], "policies": [{"id": "rule", "available": True}], "device": "cpu"},
    }

    def route_api(route):
        path = route.request.url.split("/api/", 1)[1]
        if path == "catalog":
            route.fulfill(json=catalog)
            return
        if path == "profiles/catalog":
            route.fulfill(json={"models": [model, {**model, "id": "sft-202", "seed": 202}], "default_model": model["id"], "results": {"status": {"stage": "completed"}}})
            return
        body = route.request.post_data_json
        calls.append(body)
        if body["action"] == "start":
            state = {"id": "test-chat", "owner_token": "private-browser-token", "model": model,
                     "revision": 0, "visit": 1, "date": "2038-06-12", "context": None,
                     "question_budget": 2, "questions_remaining": 2, "messages": [],
                     "past_visits": [], "memory": [], "pending": None}
            chats[state["id"]] = state
            route.fulfill(json=state)
            return
        state = chats[body["id"]]
        assert body["owner_token"] == "private-browser-token"
        if body["action"] == "send":
            state["revision"] += 1
            state["messages"] = [
                {"role": "user", "text": body["text"]},
                {"role": "assistant", "text": "Para esta visita, sopa de zapallo.", "trace": {"selected_action": "PROCEED"}},
            ]
            state["memory"] = [{"memory_id": "m1", "attribute": "dinner_preference", "value": "sopa de zapallo",
                                "status": "confirmed", "applicable": True,
                                "scope": {"kind": "session", "valid_from": "2038-06-12", "valid_until_exclusive": None,
                                          "context": None, "session_id": "S1"},
                                "evidence_quotes": [{"turn_id": "u1", "text": body["text"]}]}]
        elif body["action"] == "new_visit":
            state["revision"] += 1
            state["past_visits"].append({"visit": state["visit"], "date": state["date"], "messages": state["messages"]})
            state["visit"] += 1
            state["date"], state["context"], state["messages"] = body["date"], body["context"], []
            state["memory"][0]["applicable"] = False
        response = {key: deepcopy(value) for key, value in state.items() if key != "owner_token"}
        route.fulfill(json=response)

    page.route("**/api/**", route_api)
    page.goto(url)
    playwright.expect(page.locator('#profiles-model option[value="sft-202"]')).to_have_text("SFT · semilla 202 · época 1")
    page.locator("#profiles-model").select_option("sft-202")
    playwright.expect(page.locator("#profiles-model")).to_have_value("sft-202")
    playwright.expect(page.locator("#profiles-model")).to_be_enabled()
    page.locator("#profiles-model").select_option("sft-101")
    page.locator("#profiles-message").fill("Solo para esta sesión, sopa de zapallo. <script>alert('texto')</script>")
    page.get_by_role("button", name="Enviar mensaje", exact=True).click()
    playwright.expect(page.locator(".profiles-record")).to_contain_text("sopa de zapallo")
    page.reload()
    playwright.expect(page.locator("#profiles .chat-messages")).to_contain_text("<script>alert('texto')</script>")
    assert calls[-1]["action"] == "load"
    assert calls[-1]["owner_token"] == "private-browser-token"
    saved_menu = page.get_by_label("Conversaciones guardadas en este navegador")
    playwright.expect(saved_menu).to_be_visible()
    assert "private-browser-token" not in saved_menu.inner_html()
    page.locator("#profiles-model").select_option("sft-202")
    playwright.expect(page.locator(".profiles-record")).to_have_count(0)
    saved_menu.select_option(index=1)
    playwright.expect(page.locator("#profiles-model")).to_have_value("sft-101")
    playwright.expect(page.locator(".profiles-record")).to_contain_text("sopa de zapallo")
    assert calls[-1]["action"] == "load"

    page.get_by_text("Cambiar de visita o fecha", exact=True).click()
    page.locator("#profiles-date").fill("2038-06-15")
    page.get_by_role("button", name="Saltar a la siguiente visita", exact=True).click()
    playwright.expect(page.locator(".profiles-visit")).to_contain_text("Visita 2")
    playwright.expect(page.locator(".profiles-record")).to_contain_text("Vencido")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    for width in (768, 1024, 1440):
        page.set_viewport_size({"width": width, "height": 900})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert not errors
    assert [call["action"] for call in calls].count("start") == 1
    page.close()


def test_contextual_memory_requires_place_and_results_wait_for_verification(profiles_browser):
    browser, url = profiles_browser
    page = browser.new_page()
    page.goto(url.split("#", 1)[0])
    text = page.evaluate("""async () => {
      const {memoryRecord} = await import('/static/profiles-memory.js');
      const state = {date: '2038-06-12', visit: 1, context: null};
      const record = {memory_id: 'm1', attribute: 'drink_preference', value: 'agua de coco',
        status: 'confirmed', applicable: null, evidence_quotes: [],
        scope: {kind: 'contextual', valid_from: '2038-06-01', valid_until_exclusive: null,
          context: 'oficina', session_id: null}};
      return memoryRecord(record, state).textContent;
    }""")
    assert "Requiere indicar el lugar" in text
    assert "Vigente" not in text
    table_counts = page.evaluate("""async () => {
      const {renderProfileStudy} = await import('/static/profiles-study.js');
      const container = document.createElement('section');
      const results = {reserved: {selection_verified: false, panels: {validation_selected: {
        summary: {SFT: {macro: {longitudinal: 1.2, current_supported: .9, future_supported: .8,
          questions: 1, future_unsupported_reuse: .1}}}, comparisons: {'SFT-ramas': {passes: false,
          contrasts: {longitudinal: {estimate: -.03, interval: [-.1, .02]}}}}}},
        semantic: {parsers: {v5: {visits: 54, matches: {response_exact: 51, new_write_exact: 50, retrieval_exact: 52}}}}}};
      renderProfileStudy(container, results);
      const before = container.querySelectorAll('table').length;
      results.reserved.selection_verified = true;
      renderProfileStudy(container, results);
      return {before, tables: container.querySelectorAll('table').length, text: container.textContent};
    }""")
    assert table_counts["before"] == 0 and table_counts["tables"] == 2
    assert "No alcanza el criterio predefinido" in table_counts["text"]
    assert "51/54" in table_counts["text"] and "GRPO T4" in table_counts["text"]
    page.close()
