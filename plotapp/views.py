import pandas as pd
import numpy as np
from django.shortcuts import render
from .forms import PlantParametersForm
import pyomo.environ as pyo
import tempfile
import matplotlib.pyplot as plt
import io
import base64

def upload_view(request):
    if request.method == "POST":
        form = PlantParametersForm(request.POST, request.FILES)

        if form.is_valid():
            # --- Read Excel file ---
            file = request.FILES["file"]
            dam_euro = pd.read_excel(file)

            bgn_euro_rate = 1.95583
            dam_bgn = dam_euro.values * bgn_euro_rate

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

            # --- Create PYOMO model ---
            n_hours = 365 * 24
            T = range(n_hours)
            model = pyo.ConcreteModel()
            model.hour = pyo.RangeSet(0, n_hours - 1)

            # Variables
            model.u = pyo.Var(model.hour, within=pyo.Binary)
            model.v = pyo.Var(model.hour, within=pyo.Binary)
            model.p = pyo.Var(model.hour, within=pyo.NonNegativeReals)

            # Constraints
            model.gen_limit_upper = pyo.Constraint(model.hour, rule=lambda m,h: m.p[h] <= max_power * m.u[h])
            model.gen_limit_lower = pyo.Constraint(model.hour, rule=lambda m,h: m.p[h] >= min_power * m.u[h])
            model.ramp_up = pyo.Constraint(model.hour, rule=lambda m,h: pyo.Constraint.Skip if h == 0 else m.p[h] - m.p[h-1] <= ramp_up + max_power*(m.u[h] - m.u[h-1]))
            model.ramp_down = pyo.Constraint(model.hour, rule=lambda m,h: pyo.Constraint.Skip if h == 0 else m.p[h-1] - m.p[h] <= ramp_down + max_power*(m.u[h-1] - m.u[h]))
            model.startup_logic = pyo.Constraint(model.hour, rule=lambda m,h: m.v[h] >= m.u[h] - (m.u[h-1] if h>0 else 0))
            model.max_startups_constraint = pyo.Constraint(rule=lambda m: sum(m.v[t] for t in T) <= max_startups)
            model.min_cumulative_power = pyo.Constraint(rule=lambda m: sum(m.p[t] for t in T) >= min_cumulative_power)

            # Objective
            def obj_rule(m):
                revenue = sum(m.p[t] * market_price[t] for t in T)
                gen_cost = sum((coal_price * heat_rate + co2_price_bgn * emissions) * m.p[t] for t in T)
                startup_costs = sum(m.v[t] * startup_cost for t in T)
                return revenue - gen_cost - startup_costs

            model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

            # Solve
            solver = pyo.SolverFactory("cbc")
            solver.solve(model)

            power = [pyo.value(model.p[t]) for t in T]

            # Plot result
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(T, market_price, label="Market Price (BGN/MWh)")
            ax.step(T, power, where='mid', label="Power Output (MW)")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Value")
            ax.set_title("Unit Commitment Result")
            ax.legend()
            ax.grid(True)

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png).decode("utf-8")

            return render(request, "plotapp/result.html", {"graphic": graphic})

    else:
        form = PlantParametersForm()

    return render(request, "plotapp/upload.html", {"form": form})
