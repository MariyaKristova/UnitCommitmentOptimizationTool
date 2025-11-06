import pandas as pd
from django.shortcuts import render
from .forms import PlantParametersForm
import pyomo.environ as pyo
import matplotlib
matplotlib.use('Agg')  # for servers with no GUI
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def upload_view(request):
    if request.method == "POST":
        form = PlantParametersForm(request.POST, request.FILES)

        if form.is_valid():
            # --- Read Excel file ---
            file = request.FILES["file"]
            dam_euro = pd.read_excel(file)

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

            # --- Plot ---
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

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            image_base64 = base64.b64encode(image_png).decode("utf-8")

            return render(request, "plotapp/result.html", {"image": image_base64, "financials": financials})

    else:
        form = PlantParametersForm()

    return render(request, "plotapp/upload.html", {"form": form})
