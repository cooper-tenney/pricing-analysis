# Modern Blue UI Redesign

Redesigns the localhost web UI to look modern, sleek, and blue-toned. No changes to matplotlib PDF report generation.

## How to view

1. Run the server:
   ```bash
   uvicorn app:app --reload --port 8001
   ```

2. Open http://127.0.0.1:8001 in your browser

3. Upload a CSV (e.g. from `uploads/` folder) and click **Run pipeline**

4. View the results page with the new dashboard layout

## Changes

- **Design system**: Single `static/styles.css` with blue palette, severity colors (0 gray, 1 amber, 2 orange, 3 red), modern typography
- **Results page**: Top nav, config card (two columns), column audit with KPI tiles and worst-columns table
- **Tables**: Sticky headers, zebra striping, sortable (severity, missing_pct, nunique), search filter
- **Expandable rows**: Click chevron to reveal full flags, reasons, and stats
- **Error handling**: Run failures redirect to results with error in red alert panel instead of 500 JSON

## Screenshots

_Add screenshots of: index page, results page with audit table, expanded row_
