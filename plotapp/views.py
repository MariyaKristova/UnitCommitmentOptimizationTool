import os
import re
import zipfile
from io import StringIO, BytesIO
import csv
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from django.conf import settings
from .forms import PlantParametersForm, ExtractPeriodForm
from django.http import Http404, FileResponse, HttpResponse
from django.shortcuts import render, redirect
import matplotlib
matplotlib.use('Agg')


# utility functions
def read_excel_file(excel_filename):
    excel_path = os.path.join(settings.DATA_INPUT_DIR, excel_filename)
    df = pd.read_excel(excel_path)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    bgn_euro_rate = 1.95583
    df['Price'] = df['Price'] * bgn_euro_rate
    market_price = df['Price'].to_numpy()

    return df, market_price


def calculate_degradation(n_hours):
    if n_hours == 365 * 24:
        month_lengths = [31,28,31,30,31,30,31,31,30,31,30,31]
    else:
        month_lengths = [31,29,31,30,31,30,31,31,30,31,30,31]

    degradation = np.zeros(n_hours)
    start_idx = 0
    for idx, month_length in enumerate(month_lengths):
        stop_idx = start_idx + month_length * 24
        if stop_idx > n_hours:
            stop_idx = n_hours
        degradation[start_idx:stop_idx] = 1.071 + 0.0002 * idx
        start_idx = stop_idx

    return degradation


def build_model(n_hours, market_price, params, degradation):
    T = range(n_hours)
    model = pyo.ConcreteModel()
    model.hour = pyo.RangeSet(0, n_hours - 1)

    # variables
    model.u = pyo.Var(model.hour, within=pyo.Binary)
    model.v = pyo.Var(model.hour, within=pyo.Binary)
    model.p = pyo.Var(model.hour, within=pyo.NonNegativeReals)

    # constraints
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

    # objective
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


def compute_financials(power, commitment, startups, market_price, params, degradation):
    hours = len(power)
    revenue = sum(power[t] * market_price[t] for t in range(hours))
    gen_cost = sum(
        (params["coal_price"] * params["heat_rate"] * degradation[t] + params["co2_price_bgn"] * params["emissions"]) * power[t] for t in
        range(hours))
    startup_total = sum(startups[t] * params["startup_cost"] for t in range(hours))
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


def save_results_csv(financials, load_curve_df, index_str, date_str, params):
    results_csv_filename = f"{index_str}_{date_str}_results.csv"
    load_curve_csv_filename = f"{index_str}_{date_str}_load_curve.csv"

    # save load curve
    load_curve_df.to_csv(os.path.join(settings.DATA_OUTPUT_DIR, load_curve_csv_filename), index=False)

    # save financials and params
    rows = [("metric", "value", "unit")]
    for k, v in financials.items():
        rows.append((k, v, "BGN" if "cost" in k or "revenue" in k or "profit" in k else ""))

    if params:
        rows.append(("---parameters---", "", ""))
        for k, v in params.items():
            rows.append((k, v, ""))

    with open(os.path.join(settings.DATA_OUTPUT_DIR, results_csv_filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

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

def render_result_template(request, image_base64, financials, results_csv_file, load_curve_csv_file, png_file, run_id, extract_form):
    context = {
        "image": image_base64,
        "financials": financials,
        "results_csv_file": results_csv_file,
        "load_curve_csv_file": load_curve_csv_file,
        "png_file": png_file,
        "run_id": run_id,
        "extract_form": extract_form,
    }
    return render(request, "plotapp/result.html", context)


# main view
def upload_view(request):
    if request.method == "POST":
        form = PlantParametersForm(request.POST)
        if form.is_valid():
            # read market price
            df, market_price = read_excel_file(form.cleaned_data["excel_file"])
            n_hours = len(df)

            # calculate degradation
            degradation = calculate_degradation(n_hours)

            # prepare parameters
            params = {k: form.cleaned_data[k] for k in [
                "min_power", "max_power", "ramp_up", "ramp_down", "emissions",
                "coal_price", "heat_rate", "co2_price_bgn", "startup_cost",
                "max_startups", "min_cumulative_power", "min_cumulative_uptime"
            ]}

            # build & solve model
            model, T = build_model(n_hours, market_price, params, degradation)
            model = solve_model(model)
            power, commitment, startups = extract_results(model, T)

            # financial metrics
            financials = compute_financials(power, commitment, startups, market_price, params, degradation)

            # generate run_id and date_str
            existing_files = os.listdir(settings.DATA_OUTPUT_DIR)
            numbers = [int(m.group(1)) for f in existing_files if (m := re.match(r"^(\d{4})_", f))]
            next_index = max(numbers) + 1 if numbers else 1
            run_id = str(next_index).zfill(4)
            date_str = datetime.now().strftime("%Y%m%d")

            # save load curve with DateTime
            load_curve_df = pd.DataFrame({
                'DateTime': df['DateTime'],
                "Hour": list(T),
                "Power_Output_MW": power,
                "Commitment": commitment,
                "Startups": startups,
                "Market_Price_BGN_per_MWh": market_price
            })
            results_csv_file, load_curve_csv_file = save_results_csv(financials, load_curve_df, run_id, date_str, params)

            # save plot
            png_file, image_base64 = generate_plot(T, market_price, power, commitment, params["max_power"], run_id,
                                                   date_str)

            extract_form = ExtractPeriodForm()

            return render_result_template(
                request,
                image_base64,
                financials,
                results_csv_file,
                load_curve_csv_file,
                png_file,
                run_id=run_id,
                extract_form=extract_form,
            )

    else:
        form = PlantParametersForm()

    return render(request, "plotapp/upload.html", {"form": form})


def view_result(request, run_id, extract_form=None):
    files = os.listdir(settings.DATA_OUTPUT_DIR)

    results_csv_file = next((f for f in files if f.startswith(run_id) and f.endswith("_results.csv")), None)
    load_curve_csv_file = next((f for f in files if f.startswith(run_id) and f.endswith("_load_curve.csv")), None)
    png_file = next((f for f in files if f.startswith(run_id) and f.endswith("_plot.png")), None)

    if not results_csv_file or not load_curve_csv_file or not png_file:
        raise Http404("Result files not found")

    df_results = pd.read_csv(os.path.join(settings.DATA_OUTPUT_DIR, results_csv_file))
    financials = dict(zip(df_results["metric"], df_results["value"]))

    # encode PNG for page
    png_path = os.path.join(settings.DATA_OUTPUT_DIR, png_file)
    with open(png_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # initialize extract form
    if extract_form is None:
        extract_form = ExtractPeriodForm()

    return render_result_template(
        request,
        image_base64,
        financials,
        results_csv_file,
        load_curve_csv_file,
        png_file,
        run_id=run_id,
        extract_form=extract_form
    )


def download_curve_csv(request, filename):
    # serve CSV curve file from DATA_OUTPUT_DIR
    file_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise Http404("File not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=filename)

def download_png(request, filename):
    # serve PNG file from DATA_OUTPUT_DIR
    file_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise Http404("File not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=filename)

def download_results_csv(request, filename):
    # serve CSV results file from DATA_OUTPUT_DIR
    file_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise Http404("File not found")
    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=filename)


def extracted_result_view(request, run_id):
    files = os.listdir(settings.DATA_OUTPUT_DIR)

    results_csv_file = next((f for f in files if f.startswith(run_id) and f.endswith("_results.csv")), None)
    load_curve_csv_file = next((f for f in files if f.startswith(run_id) and f.endswith("_load_curve.csv")), None)
    if not results_csv_file or not load_curve_csv_file:
        raise Http404("Required file not found")

    # read load curve
    load_curve_df = pd.read_csv(os.path.join(settings.DATA_OUTPUT_DIR, load_curve_csv_file), parse_dates=['DateTime'])

    # read params from results csv
    params = {}
    with open(os.path.join(settings.DATA_OUTPUT_DIR, results_csv_file), newline="") as f:
        reader = csv.reader(f)
        section = "metrics"
        for row in reader:
            if row and row[0] == "---parameters---":
                section = "params"
                continue
            if section == "params" and row:
                key, value = row[0], row[1]
                try:
                    params[key] = float(value)
                except:
                    params[key] = value

    extract_form = ExtractPeriodForm(request.POST or None)
    financials = {}
    image_base64 = ""
    start_date = None
    end_date = None

    if request.method == "POST":
        if extract_form.is_valid():
            start_date = extract_form.cleaned_data['start_date']
            end_date = extract_form.cleaned_data['end_date']

            year = load_curve_df['DateTime'].dt.year.iloc[0]
            start_dt = start_date.replace(year=year)
            end_dt = end_date.replace(year=year)

            mask = (load_curve_df['DateTime'] >= pd.Timestamp(start_dt)) & (load_curve_df['DateTime'] <= pd.Timestamp(end_dt))
            period_df = load_curve_df.loc[mask].copy()
            if period_df.empty:
                extract_form.add_error(None, "No data for selected period")
            else:
                # extract data for financials and plot
                T = list(range(len(period_df)))
                power = period_df["Power_Output_MW"].to_list()
                commitment = period_df["Commitment"].to_list()
                startups = period_df["Startups"].to_list()
                market_price = period_df["Market_Price_BGN_per_MWh"].to_numpy()
                degradation = np.ones(len(period_df))

                # compute financials
                financials = compute_financials(power, commitment, startups, market_price, params, degradation)

                # generate PNG in memory
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(T, market_price, label="Market Price (BGN/MWh)", color='black')
                ax.step(T, power, where='mid', label="Power Output (MW)", linewidth=2)
                ax.fill_between(T, 0, [params['max_power'] * u for u in commitment], color='lightgreen', alpha=0.3, step='mid',
                                label="Committed")
                ax.set_xlabel("Hour")
                ax.set_ylabel("Value")
                ax.set_title("Unit Commitment with Economic Dispatch (Extracted)")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()

                png_buffer = BytesIO()
                fig.savefig(png_buffer, format='png')
                plt.close(fig)
                png_buffer.seek(0)
                image_base64 = base64.b64encode(png_buffer.read()).decode("utf-8")
        else:
            return view_result(request, run_id, extract_form=extract_form)

    context = {
        "image": image_base64,
        "financials": financials,
        "run_id": run_id,
        "extract_form": extract_form,
        "start_date": start_date.strftime("%d.%m") if start_date else "",
        "end_date": end_date.strftime("%d.%m") if end_date else "",
    }

    return render(request, "plotapp/extracted_result.html", context)


def download_extracted_zip(request, run_id):
    files = os.listdir(settings.DATA_OUTPUT_DIR)

    results_csv_file = next((f for f in files if f.startswith(run_id) and f.endswith("_results.csv")), None)
    load_curve_csv_file = next((f for f in files if f.startswith(run_id) and f.endswith("_load_curve.csv")), None)
    if not results_csv_file or not load_curve_csv_file:
        raise Http404("required files not found")

    extract_form = ExtractPeriodForm(request.GET)
    if not extract_form.is_valid():
        raise Http404("invalid date range")

    start_date = extract_form.cleaned_data['start_date']
    end_date = extract_form.cleaned_data['end_date']

    # read load curve
    load_curve_df = pd.read_csv(os.path.join(settings.DATA_OUTPUT_DIR, load_curve_csv_file), parse_dates=['DateTime'])
    year = load_curve_df['DateTime'].dt.year.iloc[0]
    start_dt = start_date.replace(year=year)
    end_dt = end_date.replace(year=year)
    mask = (load_curve_df['DateTime'] >= pd.Timestamp(start_dt)) & (load_curve_df['DateTime'] <= pd.Timestamp(end_dt))
    period_df = load_curve_df.loc[mask].copy()
    if period_df.empty:
        raise Http404("no data for selected period")

    # read params from results csv
    params = {}
    with open(os.path.join(settings.DATA_OUTPUT_DIR, results_csv_file), newline="") as f:
        reader = csv.reader(f)
        section = "metrics"
        for row in reader:
            if row and row[0] == "---parameters---":
                section = "params"
                continue
            if section == "params" and row:
                key, value = row[0], row[1]
                try:
                    params[key] = float(value)
                except:
                    params[key] = value

    # extract data for financials and plot
    T = list(range(len(period_df)))
    power = period_df["Power_Output_MW"].to_list()
    commitment = period_df["Commitment"].to_list()
    startups = period_df["Startups"].to_list()
    market_price = period_df["Market_Price_BGN_per_MWh"].to_numpy()
    degradation = np.ones(len(period_df))

    financials = compute_financials(power, commitment, startups, market_price, params, degradation)

    # create in-memory zip
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # CSV of period
        period_csv_buffer = StringIO()
        period_df.to_csv(period_csv_buffer, index=False)
        zf.writestr(f"{run_id}_extracted_load_curve.csv", period_csv_buffer.getvalue())

        # Financials CSV
        fin_buffer = StringIO()
        writer = csv.writer(fin_buffer)
        writer.writerow(["metric", "value"])
        for k, v in financials.items():
            writer.writerow([k, v])
        zf.writestr(f"{run_id}_extracted_financials.csv", fin_buffer.getvalue().encode("utf-8"))

        # PNG in memory
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(T, market_price, label="Market Price (BGN/MWh)", color='black')
        ax.step(T, power, where='mid', label="Power Output (MW)", linewidth=2)
        ax.fill_between(T, 0, [params['max_power'] * u for u in commitment], color='lightgreen', alpha=0.3, step='mid',
                        label="Committed")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Value")
        ax.set_title("Unit Commitment with Economic Dispatch (Extracted)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        png_buffer = BytesIO()
        fig.savefig(png_buffer, format='png')
        plt.close(fig)
        png_buffer.seek(0)
        zf.writestr(f"{run_id}_extracted_plot.png", png_buffer.read())

    zip_buffer.seek(0)
    response = HttpResponse(zip_buffer, content_type="application/zip")
    response['Content-Disposition'] = f'attachment; filename={run_id}_extracted.zip'
    return response



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


def custom_404(request, exception):
    return render(request, "plotapp/errors/404.html", status=404)

def custom_500(request):
    return render(request, "plotapp/errors/500.html", status=500)



