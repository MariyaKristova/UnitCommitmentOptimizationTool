import os
import re

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.http import Http404, FileResponse, HttpResponseNotAllowed
from django.shortcuts import render, redirect
from .forms import PlantParametersForm
import pyomo.environ as pyo
import matplotlib
matplotlib.use('Agg')  # for servers with no GUI
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from datetime import datetime

import os
import re
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from django.conf import settings
from django.shortcuts import render
from .forms import PlantParametersForm


# --- Utility functions ---
def read_excel_file(excel_filename):
    excel_path = os.path.join(settings.DATA_INPUT_DIR, excel_filename)
    dam_euro = pd.read_excel(excel_path)
    bgn_euro_rate = 1.95583
    dam_bgn = dam_euro.values * bgn_euro_rate
    n_hours = 365 * 24
    market_price = dam_bgn[:n_hours].flatten()
    return market_price


def calculate_degradation(length):
    months = range(175, 187)
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    degradation = np.zeros(length)
    start_idx = 0
    for idx, month_length in enumerate(month_lengths):
        stop_idx = start_idx + month_length * 24
        if stop_idx > length:
            stop_idx = length
        degradation[start_idx:stop_idx] = 1.071 + 0.0002 * months[idx]
        start_idx = stop_idx
    return degradation


def build_model(n_hours, market_price, params, degradation):
    T = range(n_hours)
    model = pyo.ConcreteModel()
    model.hour = pyo.RangeSet(0, n_hours - 1)

    # Variables
    model.u = pyo.Var(model.hour, within=pyo.Binary)
    model.v = pyo.Var(model.hour, within=pyo.Binary)
    model.p = pyo.Var(model.hour, within=pyo.NonNegativeReals)

    # Constraints
    model.gen_limit_upper = pyo.Constraint(model.hour, rule=lambda m, h: m.p[h] <= params["max_power"] * m.u[h])
    model.gen_limit_lower = pyo.Constraint(model.hour, rule=lambda m, h: m.p[h] >= params["min_power"] * m.u[h])
    model.ramp_up = pyo.Constraint(model.hour,
                                   rule=lambda m, h: pyo.Constraint.Skip if h == 0 else m.p[h] - m.p[h - 1] <= params[
                                       "ramp_up"] + params["max_power"] * (m.u[h] - m.u[h - 1]))
    model.ramp_down = pyo.Constraint(model.hour,
                                     rule=lambda m, h: pyo.Constraint.Skip if h == 0 else m.p[h - 1] - m.p[h] <= params[
                                         "ramp_down"] + params["max_power"] * (m.u[h - 1] - m.u[h]))
    model.startup_logic = pyo.Constraint(model.hour, rule=lambda m, h: m.v[h] >= m.u[h] - (m.u[h - 1] if h > 0 else 0))
    model.max_startups_constraint = pyo.Constraint(rule=lambda m: sum(m.v[t] for t in T) <= params["max_startups"])
    model.min_cumulative_power = pyo.Constraint(rule=lambda m: sum(m.p[t] for t in T) >= params["min_cumulative_power"])
    model.min_cumulative_uptime = pyo.Constraint(
        rule=lambda m: sum(m.u[t] for t in T) >= params["min_cumulative_uptime"])

    # Objective
    def obj_rule(m):
        revenue = sum(m.p[t] * market_price[t] for t in T)
        gen_cost = sum((params["coal_price"] * params["heat_rate"] * degradation[t] + params["co2_price_bgn"] * params[
            "emissions"]) * m.p[t] for t in T)
        startup_costs = sum(m.v[t] * params["startup_cost"] for t in T)
        return revenue - gen_cost - startup_costs

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    return model, T


def solve_model(model):
    solver = pyo.SolverFactory("cbc")
    solver.solve(model)
    return model


def extract_results(model, T):
    power = [pyo.value(model.p[t]) for t in T]
    commitment = [pyo.value(model.u[t]) for t in T]
    startups = [pyo.value(model.v[t]) for t in T]
    return power, commitment, startups


def compute_financials(power, commitment, startups, market_price, params):
    revenue = sum(power[t] * market_price[t] for t in range(len(power)))
    gen_cost = sum(
        (params["coal_price"] * params["heat_rate"] + params["co2_price_bgn"] * params["emissions"]) * power[t] for t in
        range(len(power)))
    startup_total = sum(startups[t] * params["startup_cost"] for t in range(len(power)))
    total_profit = revenue - gen_cost - startup_total

    total_power = sum(power)
    return {
        "total_commitment_hours": sum(commitment),
        "relative_uptime_percent": (sum(commitment) / len(power)) * 100,
        "total_revenue": revenue,
        "revenue_per_MWh": revenue / total_power if total_power > 0 else 0,
        "total_profit": total_profit,
        "profit_per_MWh": total_profit / total_power if total_power > 0 else 0,
        "total_expenses": gen_cost + startup_total,
        "expenses_per_MWh": (gen_cost + startup_total) / total_power if total_power > 0 else 0,
        "coal_co2_expenses": gen_cost,
        "coal_co2_per_MWh": gen_cost / total_power if total_power > 0 else 0,
        "total_startups": sum(startups),
        "startup_cost_total": startup_total,
        "startup_cost_per_MWh": startup_total / total_power if total_power > 0 else 0
    }


def save_results_csv(financials, load_curve_df, index_str, date_str):
    results_csv_filename = f"{index_str}_{date_str}_results.csv"
    load_curve_csv_filename = f"{index_str}_{date_str}_load_curve.csv"

    # Save load curve
    load_curve_df.to_csv(os.path.join(settings.DATA_OUTPUT_DIR, load_curve_csv_filename), index=False)

    # Save financials
    import csv
    financials_csv = [("metric", "value", "unit")]
    for k, v in financials.items():
        financials_csv.append((k, v, "BGN" if "cost" in k or "revenue" in k or "profit" in k else ""))

    with open(os.path.join(settings.DATA_OUTPUT_DIR, results_csv_filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(financials_csv)

    return results_csv_filename, load_curve_csv_filename


def generate_plot(T, market_price, power, commitment, max_power, index_str, date_str):
    png_filename = f"{index_str}_{date_str}_plot.png"
    png_path = os.path.join(settings.DATA_OUTPUT_DIR, png_filename)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(T, market_price, label="Market Price (BGN/MWh)", color='black')
    ax.step(T, power, where='mid', label="Power Output (MW)", linewidth=2)
    ax.fill_between(T, 0, [max_power * u for u in commitment], color='lightgreen', alpha=0.3, step='mid',
                    label="Committed")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Value")
    ax.set_title("Unit Commitment with Economic Dispatch")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    with open(png_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return png_filename, image_base64


# --- Main view ---
def upload_view(request):
    if request.method == "POST":
        form = PlantParametersForm(request.POST)
        if form.is_valid():
            # --- Read market price ---
            market_price = read_excel_file(form.cleaned_data["excel_file"])
            degradation = calculate_degradation(len(market_price))

            # --- Prepare parameters ---
            params = {k: form.cleaned_data[k] for k in [
                "min_power", "max_power", "ramp_up", "ramp_down", "emissions",
                "coal_price", "heat_rate", "co2_price_bgn", "startup_cost",
                "max_startups", "min_cumulative_power", "min_cumulative_uptime"
            ]}

            # --- Build & solve model ---
            n_hours = len(market_price)
            model, T = build_model(n_hours, market_price, params, degradation)
            model = solve_model(model)
            power, commitment, startups = extract_results(model, T)

            # --- Financial metrics ---
            financials = compute_financials(power, commitment, startups, market_price, params)

            # --- Determine next index ---
            existing_files = os.listdir(settings.DATA_OUTPUT_DIR)
            numbers = [int(m.group(1)) for f in existing_files if (m := re.match(r"^(\d{4})_", f))]
            next_index = max(numbers) + 1 if numbers else 1
            index_str = str(next_index).zfill(4)
            date_str = datetime.now().strftime("%Y%m%d")

            # --- Save CSVs ---
            load_curve_df = pd.DataFrame({
                "Hour": list(T),
                "Power_Output_MW": power,
                "Commitment": commitment,
                "Startups": startups,
                "Market_Price_BGN_per_MWh": market_price
            })
            results_csv_file, load_curve_csv_file = save_results_csv(financials, load_curve_df, index_str, date_str)

            # --- Save plot ---
            png_file, image_base64 = generate_plot(T, market_price, power, commitment, params["max_power"], index_str,
                                                   date_str)

            return render(request, "plotapp/result.html", {
                "image": image_base64,
                "financials": financials,
                "results_csv_file": results_csv_file,
                "load_curve_csv_file": load_curve_csv_file,
                "png_file": png_file
            })
    else:
        form = PlantParametersForm()

    return render(request, "plotapp/upload.html", {"form": form})


def download_curve_csv(request, filename):
    """Serve CSV curve file from DATA_OUTPUT_DIR"""
    file_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise Http404("File not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=filename)

def download_png(request, filename):
    """Serve PNG file from DATA_OUTPUT_DIR"""
    file_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise Http404("File not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=filename)

def download_results_csv(request, filename):
    """Serve CSV results file from DATA_OUTPUT_DIR"""
    file_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise Http404("File not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=filename)

def all_results(request):
    files = os.listdir(settings.DATA_OUTPUT_DIR)
    results = []

    for f in files:
        if f.endswith("_results.csv"):
            # format: 0001_20251108_results.csv
            parts = f.split("_")  # ["0001","20251108","results.csv"]
            run_id = parts[0]
            date = parts[1]
            results.append((run_id, date))

    # sort descending by run_id
    results = sorted(results, key=lambda x: int(x[0]), reverse=True)

    return render(request, "plotapp/all_results.html", {"results": results})


def view_result(request, run_id):
    files = os.listdir(settings.DATA_OUTPUT_DIR)

    results_csv = None
    load_curve_csv = None
    png_file = None

    for f in files:
        if f.startswith(run_id) and f.endswith("_results.csv"):
            results_csv = f
        elif f.startswith(run_id) and f.endswith("_load_curve.csv"):
            load_curve_csv = f
        elif f.startswith(run_id) and f.endswith("_plot.png"):
            png_file = f

    if not results_csv or not load_curve_csv or not png_file:
        raise Http404("Result files not found")

    # full paths
    results_csv_path = os.path.join(settings.DATA_OUTPUT_DIR, results_csv)
    load_curve_csv_path = os.path.join(settings.DATA_OUTPUT_DIR, load_curve_csv)
    png_path = os.path.join(settings.DATA_OUTPUT_DIR, png_file)

    # load CSV files
    df_results = pd.read_csv(results_csv_path)
    df_load_curve = pd.read_csv(load_curve_csv_path)

    # encode PNG for page
    with open(png_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return render(request, "plotapp/view_result.html", {
        "run_id": run_id,
        "results_csv_file": results_csv,
        "load_curve_csv_file": load_curve_csv,
        "png_file": png_file,
        "image": image_base64,
        "financials_table": df_results.to_html(classes="table table-striped", index=False),
        "load_curve_table": df_load_curve.to_html(classes="table table-striped", index=False)
    })


def delete_result(request, run_id):
    files = os.listdir(settings.DATA_OUTPUT_DIR)
    deleted_files = []

    for f in files:
        if f.startswith(run_id):
            file_path = os.path.join(settings.DATA_OUTPUT_DIR, f)
            try:
                os.remove(file_path)
                deleted_files.append(f)
            except Exception as e:
                print(f"Error deleting {f}: {e}")

    if not deleted_files:
        raise Http404("No files found to delete for this run_id")

    return redirect("all_results")