# Frontend Changes: Theme Toggle Button

Adds a sun/moon toggle button in the top-right corner that switches between the existing dark theme and a new light theme.

## `frontend/index.html`
- Added `<button id="themeToggle" class="theme-toggle">` at the top of `<body>`, containing two inline SVG icons (`.icon-sun`, `.icon-moon`) marked `aria-hidden="true"`.
- Added a small inline script in `<head>` that reads `localStorage.theme` and sets `data-theme="light"` on `<html>` before first paint, so there is no flash of the wrong theme on reload.
- Bumped cache-busting versions: `style.css?v=11`, `script.js?v=10`.

## `frontend/style.css`
- New CSS variables in `:root` (dark, default): `--link-color`, `--code-bg`, `--toggle-bg`, `--toggle-icon`.
- Replaced hardcoded colors with those variables (source-chip link color, inline/block code background). Fixed the blockquote border, which referenced an undefined `--primary` variable, to use `--primary-color`.
- New `[data-theme="light"]` block overriding all color variables for a light palette.
- Smooth 0.3s background/border/text color transitions on the main surfaces when switching themes.
- `.theme-toggle` styles: fixed at top-right, 44px circular button matching the existing surface/border/shadow look, hover lift and blue glow like the send button, `:focus-visible` focus ring using `--focus-ring`.
- Icon animation: the active icon is visible; the inactive one is rotated ±90° and scaled to 0.5 with opacity 0. Toggling rotates and cross-fades them (0.4s).
- `prefers-reduced-motion: reduce` disables the toggle animations.
- Responsive: the button is slightly smaller on mobile, and `.chat-messages` gets extra top padding on screens ≤1280px so the fixed button doesn't cover the first message.

## `frontend/script.js`
- `toggleTheme()` switches the `data-theme` attribute on `<html>` and saves the choice to `localStorage`. If storage isn't available, the theme still changes for the current page.
- `updateThemeToggleLabel()` keeps `aria-label`/`title` in sync with the current action ("Switch to light theme" / "Switch to dark theme").
- The click handler is registered in `setupEventListeners()`.

---

# Frontend Changes: Light Theme CSS Variables

Refined the `[data-theme="light"]` palette in `frontend/style.css` and routed every remaining hardcoded color through theme variables, so the light theme is fully consistent and meets WCAG AA.

## New variables (defined in both `:root` and `[data-theme="light"]`)
| Variable | Dark (unchanged look) | Light |
|---|---|---|
| `--input-border` | `#334155` | `#8391a7` |
| `--welcome-shadow` | `rgba(0,0,0,0.2)` shadow | `rgba(15,23,42,0.08)` shadow |
| `--error-text` / `--error-bg` / `--error-border` | `#f87171` / red 10% / red 20% | `#b91c1c` / red 8% / `rgba(185,28,28,0.3)` |
| `--success-text` / `--success-bg` / `--success-border` | `#4ade80` / green 10% / green 20% | `#15803d` / green 10% / `rgba(21,128,61,0.3)` |

## Light palette adjustments
- `--border-color`: `#e2e8f0` → `#cbd5e1` (dividers and cards were barely visible at 1.2:1).
- `--shadow` and `--focus-ring` are slightly stronger so they show up on light surfaces.
- Kept `--background #f8fafc`, `--surface #ffffff`, `--text-primary #0f172a`, `--text-secondary #475569`, `--primary-color #2563eb`, `--primary-hover #1d4ed8`, and `--link-color #1d4ed8`. All of them already pass AA or better; see the ratios below.

## Rules now using variables
- `.error-message` and `.success-message`: the old `#f87171` / `#4ade80` text had only 2.8:1 and 1.7:1 contrast on light backgrounds.
- The welcome message `box-shadow`.
- The `#chatInput` border now uses `--input-border`, so the text field's boundary meets WCAG 1.4.11 (≥3:1) in light mode.

## Verified contrast (light theme, WCAG 2.1)
- Body text `#0f172a` on surface/background: 17.9 / 17.1 (AAA)
- Secondary text `#475569`: 7.6 / 7.2 (AAA)
- Primary `#2563eb` as text: 5.2 on surface, 4.9 on background, 4.7 on hover surface (AA)
- White on user bubble `#2563eb`: 5.2 (AA)
- Source-link text `#1d4ed8` on its tinted chip: 5.7 (AA)
- Error `#b91c1c`: 5.7; success `#15803d`: 4.6 (AA)
- Input border `#8391a7`: 3.2 / 3.05 (meets the 3:1 non-text minimum)

The ratios are listed in a comment above the light-theme block in `style.css`. The dark theme looks the same as before, because its new variables hold the previous hardcoded values.

---

# Frontend Changes: Theme Switching JS & Transitions

Checked the implementation against the spec: the button toggles the theme on click, colors come from CSS custom properties, and the `data-theme` attribute is on `<html>`. The one gap was the smooth transition, which only covered a hand-picked list of elements.

## `frontend/style.css`
- Removed the hand-picked list of transitioned selectors (`body`, `.sidebar`, `.message-content`, …). Elements outside that list, such as sidebar headers, course titles, source chips, the sources toggle, and error/success messages, used to change color instantly.
- Added an `html.theme-transition` rule that gives every element and pseudo-element a 0.3s `background-color` / `border-color` / `color` / `fill` / `stroke` / `box-shadow` transition. It's `!important` so it also applies to elements that already have their own `transition` (for example `#chatInput`).
- Added an exemption so the toggle's sun/moon rotate-and-fade animation still runs while `.theme-transition` is active.
- The reduced-motion media query now also disables the icon animation while `.theme-transition` is active.

## `frontend/script.js`
- `toggleTheme()` adds `theme-transition` to `<html>` right before switching `data-theme` and removes it after 350 ms. A shared timer handles rapid repeated clicks. This way hover/focus transitions elsewhere stay the same outside a theme switch.
- When the user has `prefers-reduced-motion: reduce` set, the class is skipped and the switch is instant.

## `frontend/index.html`
- Bumped the cache-busting versions to `style.css?v=12` and `script.js?v=11`.

---

## Accessibility (theme toggle)
- The toggle is a native `<button type="button">`, so it's in the tab order and Enter/Space activate it without extra key handling.
- It has a descriptive, state-aware `aria-label`, and the decorative SVGs are hidden from assistive tech.
- It shows a visible focus ring for keyboard users, and reduced-motion preferences are respected.
