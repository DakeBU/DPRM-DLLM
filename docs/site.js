function activate(buttons, panels, buttonKey, panelKey, value) {
  buttons.forEach((button) => {
    const active = button.dataset[buttonKey] === value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  panels.forEach((panel) => {
    const active = panel.dataset[panelKey] === value;
    panel.classList.toggle("active", active);
    panel.hidden = !active;
  });
}

function bindTabs(buttonSelector, panelSelector, buttonKey, panelKey, workspaceSelector) {
  const buttons = [...document.querySelectorAll(buttonSelector)];
  const panels = [...document.querySelectorAll(panelSelector)];
  buttons.forEach((button, index) => {
    button.addEventListener("click", () => {
      activate(buttons, panels, buttonKey, panelKey, button.dataset[buttonKey]);
      const workspace = workspaceSelector ? document.querySelector(workspaceSelector) : null;
      if (workspace) workspace.scrollTo({ left: 0, top: 0 });
    });
    button.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const next = buttons[(index + offset + buttons.length) % buttons.length];
      next.focus();
      next.click();
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindTabs(".case-tab", ".case-panel", "case", "panel", ".visual-workspace");
  bindTabs(".omni-gallery-tab", ".omni-gallery-panel", "omniGallery", "omniPanel", ".omni-gallery-workspace");
  bindTabs(".llada-gallery-tab", ".llada-gallery-panel", "lladaGallery", "lladaPanel", ".llada-gallery-workspace");
  bindTabs(".host-tab", ".host-panel", "host", "hostPanel");
  if (window.lucide) window.lucide.createIcons();
});
