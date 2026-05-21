from __future__ import annotations


APP_CSS = """
:root {
  color-scheme: dark;
  --notes-bg: #171611;
  --notes-bg-2: #211f19;
  --notes-sidebar: rgba(33, 31, 25, 0.9);
  --notes-paper: #24221c;
  --notes-paper-warm: #332d1d;
  --notes-yellow: #ffd65a;
  --notes-yellow-soft: rgba(255, 214, 90, 0.18);
  --notes-ink: #f4f1e8;
  --notes-muted: #b5ae9c;
  --notes-line: rgba(255, 237, 190, 0.14);
  --notes-shadow: 0 18px 46px rgba(0, 0, 0, 0.42);
  --notes-radius: 16px;
  --notes-card: rgba(39, 37, 30, 0.94);
  --notes-card-2: rgba(46, 42, 31, 0.92);
  --notes-field: #1f1d18;
  --notes-elevated: rgba(28, 27, 23, 0.86);
}

.gradio-container {
  max-width: 100% !important;
  background:
    radial-gradient(circle at 18% 0%, rgba(255, 214, 90, 0.10), transparent 28%),
    linear-gradient(180deg, #191813 0%, #12110e 100%) !important;
  color: var(--notes-ink) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", "Noto Sans JP", sans-serif !important;
}

.gradio-container * {
  letter-spacing: 0 !important;
}

#notes-root {
  max-width: 1720px;
  margin: 0 auto;
}

.top-toolbar {
  position: sticky;
  top: 0;
  z-index: 20;
  padding: 12px 14px;
  border: 1px solid var(--notes-line);
  border-radius: 18px;
  background: rgba(31, 29, 23, 0.88);
  box-shadow: var(--notes-shadow);
  backdrop-filter: blur(18px);
}

.toolbar-title {
  font-size: 24px;
  font-weight: 780;
  margin: 0;
}

.toolbar-subtitle {
  color: var(--notes-muted);
  margin-top: 2px;
  font-size: 13px;
}

.workspace-grid {
  min-height: calc(100vh - 210px);
  gap: 14px !important;
}

.sidebar-shell,
.note-list-shell,
.detail-shell {
  border: 1px solid var(--notes-line);
  border-radius: 18px;
  background: var(--notes-elevated);
  box-shadow: var(--notes-shadow);
  overflow: hidden;
}

.sidebar {
  min-height: 760px;
  max-height: calc(100vh - 190px);
  overflow-y: auto;
  padding: 18px 14px;
  background: var(--notes-sidebar);
}

.sidebar-brand {
  padding: 6px 6px 16px;
}

.sidebar-title {
  font-size: 22px;
  font-weight: 760;
}

.sidebar-subtitle,
.sidebar-heading,
.note-meta,
.paper-kicker {
  color: var(--notes-muted);
  font-size: 12px;
}

.sidebar-section {
  margin: 14px 0;
}

.sidebar-heading {
  font-weight: 680;
  text-transform: uppercase;
  margin: 10px 8px 8px;
}

.sidebar-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 32px;
  padding: 8px 10px;
  border-radius: 12px;
  color: var(--notes-ink);
  font-size: 14px;
}

.sidebar-item.active {
  background: var(--notes-yellow-soft);
  box-shadow: inset 0 0 0 1px rgba(255, 214, 90, 0.32);
  font-weight: 700;
}

.sidebar-item.compact b {
  color: var(--notes-muted);
  font-size: 12px;
}

.note-list-toolbar {
  padding: 12px 14px;
  border-bottom: 1px solid var(--notes-line);
  background: rgba(24, 23, 19, 0.72);
}

.scope-selector input[type="radio"],
.note-filter-toggle input[type="checkbox"],
.note-list-toolbar input[type="checkbox"],
.sidebar-shell input[type="radio"] {
  accent-color: var(--notes-yellow) !important;
  filter: drop-shadow(0 0 6px rgba(255, 214, 90, 0.28));
}

.scope-selector label,
.note-filter-toggle label {
  border-radius: 12px !important;
  border: 1px solid transparent !important;
  transition: background 140ms ease, border-color 140ms ease, color 140ms ease, box-shadow 140ms ease;
}

.scope-selector label:hover,
.note-filter-toggle label:hover {
  background: rgba(255, 240, 198, 0.07) !important;
  border-color: rgba(255, 237, 190, 0.12) !important;
}

.scope-selector label:has(input[type="radio"]:checked),
.note-filter-toggle label:has(input[type="checkbox"]:checked) {
  background: var(--notes-yellow-soft) !important;
  border-color: rgba(255, 214, 90, 0.40) !important;
  color: var(--notes-ink) !important;
  box-shadow: inset 3px 0 0 var(--notes-yellow), 0 6px 16px rgba(0, 0, 0, 0.20);
  font-weight: 720 !important;
}

.scope-selector label:has(input[type="radio"]:checked) input,
.note-filter-toggle label:has(input[type="checkbox"]:checked) input {
  outline: 2px solid rgba(255, 214, 90, 0.58);
  outline-offset: 2px;
}

.note-list {
  max-height: calc(100vh - 330px);
  overflow-y: auto;
  padding: 12px;
}

.suggestion-list {
  max-height: calc(100vh - 330px);
  overflow-y: auto;
  padding: 12px;
  scrollbar-gutter: stable;
}

.suggestion-list .suggestion-card {
  margin-right: 4px;
}

.note-card {
  margin: 0 0 10px;
  padding: 14px;
  border: 1px solid rgba(255, 237, 190, 0.12);
  border-radius: 16px;
  background: var(--notes-card);
  box-shadow: 0 10px 24px rgba(0,0,0,0.22);
  cursor: pointer;
  transition: background 140ms ease, border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
}

.note-card:hover {
  background: rgba(51, 47, 36, 0.96);
  border-color: rgba(255, 214, 90, 0.28);
  transform: translateY(-1px);
}

.note-card:focus-visible {
  outline: 2px solid rgba(255, 214, 90, 0.82);
  outline-offset: 2px;
}

.note-card.selected {
  background: var(--notes-paper-warm);
  border-color: rgba(255, 214, 90, 0.44);
  box-shadow: 0 12px 30px rgba(0,0,0,0.30), inset 3px 0 0 var(--notes-yellow);
}

.note-title {
  font-size: 16px;
  line-height: 1.36;
  font-weight: 760;
}

.note-snippet {
  margin-top: 7px;
  color: #d5cdbc;
  line-height: 1.5;
  font-size: 13px;
}

.note-meta {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 8px;
}

.note-card-footer {
  margin-top: 10px;
}

.badge-row,
.metric-row,
.status-wrap {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}

.note-badge,
.status-badge,
.confidence-pill,
.importance-pill,
.score-pill {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(255, 240, 198, 0.10);
  border: 1px solid rgba(255, 237, 190, 0.14);
  color: #e7ddc7;
  font-size: 12px;
  white-space: nowrap;
}

.note-badge.important,
.importance-pill.high {
  background: rgba(255, 214, 90, 0.24);
  color: #ffe39a;
}

.note-badge.review,
.confidence-pill.low,
.status-badge.warn {
  background: rgba(255, 133, 73, 0.18);
  color: #ffbd96;
}

.status-badge.ok,
.confidence-pill.ok,
.importance-pill.ok {
  background: rgba(142, 211, 132, 0.18);
  color: #bce9ae;
}

.muted {
  color: var(--notes-muted);
}

.detail-pane {
  max-height: calc(100vh - 205px);
  overflow-y: auto;
  padding: 16px;
  background: linear-gradient(180deg, rgba(29, 28, 24, .92), rgba(18,17,14,.98));
}

.paper {
  max-width: 860px;
  margin: 0 auto;
  padding: 30px 34px;
  border-radius: 18px;
  background: var(--notes-paper);
  border: 1px solid rgba(255, 237, 190, 0.12);
  box-shadow: 0 22px 58px rgba(0, 0, 0, 0.36);
}

.paper-title {
  font-size: clamp(24px, 3vw, 42px);
  line-height: 1.16;
  margin: 8px 0 6px;
}

.detail-meta {
  padding-bottom: 16px;
  border-bottom: 1px solid var(--notes-line);
}

.paper-section,
.ai-summary-card,
.event-card,
.thought-card,
.analysis-health {
  margin: 18px 0;
  padding: 16px;
  border: 1px solid rgba(255, 237, 190, 0.12);
  border-radius: 16px;
  background: var(--notes-card-2);
}

.month-item-card.low-priority {
  opacity: .82;
  border-style: dashed;
  background: rgba(86, 74, 56, .30);
}

.month-item-card.low-priority .note-badge {
  background: rgba(255, 198, 92, .12);
  color: #d2bb88;
}

.low-priority-details summary {
  cursor: pointer;
  font-weight: 700;
  color: #d8c58a;
}

.low-priority-details[open] summary {
  margin-bottom: 12px;
}

.section-title-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.section-title-row h2,
.section-title-row h3 {
  margin: 0 0 8px;
}

.summary-line {
  font-size: 17px;
  line-height: 1.7;
  font-weight: 620;
}

.important-points {
  padding-left: 22px;
  line-height: 1.7;
}

.revisit {
  color: #d8c58a;
  font-style: italic;
}

.evidence-card {
  margin-top: 12px;
  padding: 12px;
  border-radius: 14px;
  background: rgba(18, 17, 14, .48);
  border-left: 4px solid var(--notes-yellow);
}

.evidence-card.warning {
  border-left-color: #d36f32;
}

.evidence-label {
  font-size: 12px;
  color: var(--notes-muted);
  text-transform: uppercase;
  font-weight: 700;
}

.evidence-card blockquote {
  margin: 8px 0 0;
  padding: 0;
  line-height: 1.55;
}

.evidence-card blockquote span {
  display: inline-block;
  margin-right: 8px;
  color: var(--notes-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}

.paper-body {
  max-height: 520px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.75;
  font-size: 15px;
  padding: 18px;
  border-radius: 14px;
  color: var(--notes-ink);
  background: linear-gradient(180deg, #2a271f, #211f19);
  border: 1px solid var(--notes-line);
}

.warning-banner,
.empty-state {
  margin: 10px 0;
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255, 190, 94, 0.14);
  color: #ffd9a0;
  border: 1px solid rgba(255, 190, 94, .24);
}

.empty-state {
  background: rgba(255,255,255,.06);
  color: var(--notes-muted);
}

.health-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.meter-track {
  height: 7px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255, 237, 190, .13);
}

.meter-fill {
  height: 100%;
  background: linear-gradient(90deg, #ffd95a, #8ccf73);
}

.model-run-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.model-run-table td {
  border-top: 1px solid var(--notes-line);
  padding: 7px 6px;
}

button,
input,
textarea,
select,
.wrap,
.block {
  border-radius: 12px !important;
}

input,
textarea,
select,
.gr-text-input,
.gr-textbox,
.wrap,
.block {
  background: var(--notes-field) !important;
  color: var(--notes-ink) !important;
  border-color: var(--notes-line) !important;
}

label,
.prose,
.markdown,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container li {
  color: var(--notes-ink) !important;
}

button.primary,
.primary button {
  background: var(--notes-yellow) !important;
  color: #1d1b14 !important;
  border: 1px solid rgba(187, 136, 12, 0.28) !important;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
  outline: 3px solid rgba(255, 197, 39, 0.55) !important;
  outline-offset: 2px !important;
}

@media (max-width: 1100px) {
  .workspace-grid {
    flex-direction: column;
  }
  .sidebar,
  .note-list,
  .detail-pane {
    max-height: none;
  }
  .paper {
    padding: 22px;
  }
}
"""
