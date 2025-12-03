(function () {
  const forceLight = () => {
    localStorage.setItem("theme", "light");
    document.body.dataset.theme = "light";
  };

  forceLight();
  document.addEventListener("DOMContentLoaded", () => {
    forceLight();
    document.querySelectorAll(".theme-toggle-container, .theme-toggle").forEach(el => el.remove());
  });
})();