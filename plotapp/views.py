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

def upload_view(request):
    if request.method == "POST":
        form = PlantParametersForm(request.POST)

        if form.is_valid():
            # --- Read Excel file ---
            excel_file_name = form.cleaned_data["excel_file"]
            excel_path = os.path.join(settings.DATA_INPUT_DIR, excel_file_name)
            dam_euro = pd.read_excel(excel_path)

            # Convert EUR/MWh → BGN/MWh
            bgn_euro_rate = 1.95583
            dam_bgn = dam_euro.values * bgn_euro_rate

            # Take only 1 year of hourly prices
            years = 1
            period = years * 365
            market_price = dam_bgn[:period*24].flatten()

            # --- Read plant parameters from form ---
            min_power = form.cleaned_data["min_power"]
            max_power = form.cleaned_data["max_power"]
            ramp_up = form.cleaned_data["ramp_up"]
            ramp_down = form.cleaned_data["ramp_down"]
            emissions = form.cleaned_data["emissions"]
            coal_price = form.cleaned_data["coal_price"]
            heat_rate = form.cleaned_data["heat_rate"]
            co2_price_bgn = form.cleaned_data["co2_price_bgn"]
            startup_cost = form.cleaned_data["startup_cost"]
            max_startups = form.cleaned_data["max_startups"]
            min_cumulative_power = form.cleaned_data["min_cumulative_power"]
            min_cumulative_uptime = form.cleaned_data["min_cumulative_uptime"]

            # --- Degradation ---
            months = range(175, 187)
            month_lengths = [31,28,31,30,31,30,31,31,30,31,30,31]
            degradation = np.zeros(len(market_price))
            start_idx = 0
            for idx, month_length in enumerate(month_lengths):
                stop_idx = start_idx + month_length*24
                if stop_idx > len(market_price):
                    stop_idx = len(market_price)
                degradation[start_idx:stop_idx] = 1.071 + 0.0002 * months[idx]
                start_idx = stop_idx

            # --- Create PYOMO model ---
            n_hours = 365 * 24
            T = range(n_hours)
            model = pyo.ConcreteModel()
            model.hour = pyo.RangeSet(0, n_hours - 1)

            # Variables
            model.u = pyo.Var(model.hour, within=pyo.Binary)  # On/off
            model.v = pyo.Var(model.hour, within=pyo.Binary)  # Startup indicator
            model.p = pyo.Var(model.hour, within=pyo.NonNegativeReals)  # Power MW

            # Constraints
            model.gen_limit_upper = pyo.Constraint(model.hour, rule=lambda m,h: m.p[h] <= max_power * m.u[h])
            model.gen_limit_lower = pyo.Constraint(model.hour, rule=lambda m,h: m.p[h] >= min_power * m.u[h])
            model.ramp_up = pyo.Constraint(model.hour, rule=lambda m,h: pyo.Constraint.Skip if h==0 else m.p[h]-m.p[h-1] <= ramp_up + max_power*(m.u[h]-m.u[h-1]))
            model.ramp_down = pyo.Constraint(model.hour, rule=lambda m,h: pyo.Constraint.Skip if h==0 else m.p[h-1]-m.p[h] <= ramp_down + max_power*(m.u[h-1]-m.u[h]))
            model.startup_logic = pyo.Constraint(model.hour, rule=lambda m,h: m.v[h] >= m.u[h] - (m.u[h-1] if h>0 else 0))
            model.max_startups_constraint = pyo.Constraint(rule=lambda m: sum(m.v[t] for t in T) <= max_startups)
            model.min_cumulative_power = pyo.Constraint(rule=lambda m: sum(m.p[t] for t in T) >= min_cumulative_power)
            model.min_cumulative_uptime = pyo.Constraint(rule=lambda m: sum(m.u[t] for t in T) >= min_cumulative_uptime)

            # Objective
            def obj_rule(m):
                revenue = sum(m.p[t] * market_price[t] for t in T)
                gen_cost = sum((coal_price * heat_rate * degradation[t] + co2_price_bgn * emissions) * m.p[t] for t in T)
                startup_costs = sum(m.v[t] * startup_cost for t in T)
                return revenue - gen_cost - startup_costs

            model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

            # --- Solve ---
            solver = pyo.SolverFactory("cbc")
            solver.solve(model)

            # --- Extract results ---
            power = [pyo.value(model.p[t]) for t in T]
            commitment = [pyo.value(model.u[t]) for t in T]
            startups = [pyo.value(model.v[t]) for t in T]

            # --- Financial metrics ---
            revenue = sum(power[t] * market_price[t] for t in T)
            gen_cost = sum((coal_price * heat_rate * degradation[t] + co2_price_bgn * emissions) * power[t] for t in T)
            startup_total = sum(startups[t] * startup_cost for t in T)
            total_profit = revenue - gen_cost - startup_total

            financials = {
                "total_commitment_hours": sum(commitment),
                "relative_uptime_percent": (sum(commitment)/n_hours)*100,
                "total_revenue": revenue,
                "revenue_per_MWh": revenue/sum(power) if sum(power)>0 else 0,
                "total_profit": total_profit,
                "profit_per_MWh": total_profit/sum(power) if sum(power)>0 else 0,
                "total_expenses": gen_cost+startup_total,
                "expenses_per_MWh": (gen_cost+startup_total)/sum(power) if sum(power)>0 else 0,
                "coal_co2_expenses": gen_cost,
                "coal_co2_per_MWh": gen_cost/sum(power) if sum(power)>0 else 0,
                "total_startups": sum(startups),
                "startup_cost_total": sum(startups)*startup_cost,
                "startup_cost_per_MWh": (sum(startups)*startup_cost)/sum(power) if sum(power)>0 else 0
            }

            # --- Generate next file index and names ---
            # List existing result files
            existing_files = os.listdir(settings.DATA_OUTPUT_DIR)

            # Extract existing numeric prefixes (0001, 0002, ...)
            pattern = re.compile(r"^(\d{4})_")
            numbers = []

            for fname in existing_files:
                match = pattern.match(fname)
                if match:
                    numbers.append(int(match.group(1)))

            # Determine next index
            next_index = max(numbers) + 1 if numbers else 1
            index_str = str(next_index).zfill(4)  # 0001, 0002...

            # Date stamp
            date_str = datetime.now().strftime("%Y%m%d")

            # File names
            results_csv_filename = f"{index_str}_{date_str}_results.csv"
            load_curve_csv_filename = f"{index_str}_{date_str}_load_curve.csv"
            png_filename = f"{index_str}_{date_str}_plot.png"

            results_csv_path = os.path.join(settings.DATA_OUTPUT_DIR, results_csv_filename)
            load_curve_csv_path = os.path.join(settings.DATA_OUTPUT_DIR, load_curve_csv_filename)
            png_path = os.path.join(settings.DATA_OUTPUT_DIR, png_filename)

            # --- Save load curve CSV ---
            df_load = pd.DataFrame({
                "Hour": list(T),
                "Power_Output_MW": power,
                "Commitment": commitment,
                "Startups": startups,
                "Market_Price_BGN_per_MWh": market_price
            })
            df_load.to_csv(load_curve_csv_path, index=False)

            # --- Save financials CSV ---
            import csv
            financials_csv = [
                ("metric", "value", "unit"),
                ("total_commitment_hours", financials["total_commitment_hours"], "h"),
                ("relative_uptime_percent", financials["relative_uptime_percent"], "%"),
                ("total_revenue", financials["total_revenue"], "BGN"),
                ("revenue_per_MWh", financials["revenue_per_MWh"], "BGN"),
                ("total_profit", financials["total_profit"], "BGN"),
                ("profit_per_MWh", financials["profit_per_MWh"], "BGN"),
                ("total_expenses", financials["total_expenses"], "BGN"),
                ("expenses_per_MWh", financials["expenses_per_MWh"], "BGN"),
                ("coal_co2_expenses", financials["coal_co2_expenses"], "BGN"),
                ("coal_co2_per_MWh", financials["coal_co2_per_MWh"], "BGN"),
                ("total_startups", financials["total_startups"], ""),
                ("startup_cost_total", financials["startup_cost_total"], "BGN"),
                ("startup_cost_per_MWh", financials["startup_cost_per_MWh"], "BGN"),
            ]

            with open(results_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(financials_csv)

            # --- ADDED: save PNG file to DATA_OUTPUT_DIR ---
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(T, market_price, label="Market Price (BGN/MWh)", color='black')
            ax.step(T, power, where='mid', label="Power Output (MW)", linewidth=2)
            ax.fill_between(T, 0, [max_power*u for u in commitment], color='lightgreen', alpha=0.3, step='mid', label="Committed")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Value")
            ax.set_title("Unit Commitment with Economic Dispatch")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(png_path)  # <-- saving unique PNG
            plt.close(fig)

            # --- ADDED: encode PNG for displaying on the page ---
            with open(png_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            # --- ADDED: pass file names to template for Download buttons ---
            return render(request, "plotapp/result.html", {
                "image": image_base64,
                "financials": financials,
                "results_csv_file": results_csv_filename,
                "load_curve_csv_file": load_curve_csv_filename,
                "png_file": png_filename
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
