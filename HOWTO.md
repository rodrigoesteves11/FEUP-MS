# HOWTO — Set Up and Run the Simulation (Market ABM)

A short guide to install, run, and visualize.

---

## 1) Quick Start

(Python 3.13 preferably)

```bash
pip install -r requirements.txt
python run.py
solara run app.py
```

---

## 2) What’s in this project

- run.py — main experiment (3 policies x 30 seeds; writes KPIs to results_new_model/)
- run_mixes.py — Policy x Agent mix experiment (keeps N=300)
- app.py — Solara dashboard (live + aggregates)
- extra_viz/
  - viz_simple.py — Live Matplotlib animation for a single run
  - interactive_dashboard.py — Static Plotly HTML dashboards
  - analyze_results.py — Extra analysis (boxplots, stats, report)

---

## 3) Run experiments

- Main batch (creates results_new_model/kpi_results.csv):

```bash
python run.py
```

- Policy x Mix (creates results_mixes/):

```bash
python run_mixes.py
```

---

## 4) Visualize results

- Live (Matplotlib), one run:

```bash
python extra_viz/viz_simple.py <policy> [steps] [seed]

python extra_viz/viz_simple.py none
python extra_viz/viz_simple.py moderate 100
python extra_viz/viz_simple.py excessive 75 42
```

- Solara dashboard (live + aggregates):

```bash
python run.py
solara run app.py
```

- Static HTML dashboards (Plotly):

```bash
python extra_viz/interactive_dashboard.py
```

- Extra analysis (figures + text report):

```bash
python extra_viz/analyze_results.py
```