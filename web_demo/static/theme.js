(() => {
  const key = "tesis-tema";
  const preference = window.matchMedia("(prefers-color-scheme: dark)");
  let choice;
  try { choice = localStorage.getItem(key); } catch {}
  const apply = dark => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    button.setAttribute("aria-pressed", String(dark));
    document.getElementById("theme-label").textContent = dark ? "Modo claro" : "Modo oscuro";
  };
  const current = () => choice === "dark" || (!choice && preference.matches);
  apply(current());
  preference.addEventListener("change", () => apply(current()));
  document.addEventListener("DOMContentLoaded", () => {
    apply(current());
    document.getElementById("theme-toggle").addEventListener("click", () => {
      choice = current() ? "light" : "dark";
      try { localStorage.setItem(key, choice); } catch {}
      apply(current());
    });
  });
})();
