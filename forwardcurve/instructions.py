import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo

# --- Load market price ---
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
dam_euro = pd.read_excel(file_path)

bgn_euro_rate = 1.95583
dam_bgn = dam_euro.values * bgn_euro_rate

years = 1
period = years * 365
market_price = dam_bgn[:period*24].flatten()

# --- Plant parameters ---
min_power = 270          # MW
max_power = 600          # MW
ramp_up = 100            # MW/hour
ramp_down = 100          # MW/hour
emissions = 1.447        # t CO2/MWh

offered_price = 312      # BGN
coal_price = 2.98e-6     # BGN/kJ
heat_rate = 10322e3      # kJ/MWh
co2_price_bgn = 81.56 * bgn_euro_rate

startup_cost = 49191     # Mean cost

max_startups = 33

min_cumulative_uptime = 0

min_cumulative_power = 1858758 #1858758.56

# --- Degradation ---
months = range(175, 187)
degradation = [None] * len(market_price)
month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

start_idx = 0
for idx, month_length in enumerate(month_lengths):
    stop_idx = start_idx + month_length * 24
    degradation[start_idx:stop_idx] = [1.071 + 0.0002 * months[idx]] * (stop_idx - start_idx)
    start_idx = stop_idx

# --- Model ---
n_hours = 365*24
T = range(n_hours)
model = pyo.ConcreteModel()
model.hour = pyo.RangeSet(0, n_hours-1)

# Variables
model.u = pyo.Var(model.hour, within=pyo.Binary)               # On/off
model.v = pyo.Var(model.hour, within=pyo.Binary)               # Startup
model.p = pyo.Var(model.hour, within=pyo.NonNegativeReals)    # Generation MW

# --- Constraints ---
def gen_limits_upper(m, h):
    return m.p[h] <= max_power * m.u[h]
model.gen_limit_upper = pyo.Constraint(model.hour, rule=gen_limits_upper)

def gen_limits_lower(m, h):
    return m.p[h] >= min_power * m.u[h]
model.gen_limit_lower = pyo.Constraint(model.hour, rule=gen_limits_lower)

def ramp_up_rule(m, h):
    if h == 0: return pyo.Constraint.Skip # ramp up is limited when turned on, but when it turns on it goes to max_power immediately
    return m.p[h] - m.p[h-1] <= ramp_up + max_power * (m.u[h] - m.u[h-1])
model.ramp_up = pyo.Constraint(model.hour, rule=ramp_up_rule)

def ramp_down_rule(m, h):
    if h == 0: return pyo.Constraint.Skip # ramp down is limited when turned on, but when it turns off it goes to 0 immediately
    return m.p[h-1] - m.p[h] <= ramp_down + max_power * (m.u[h-1] - m.u[h])
model.ramp_down = pyo.Constraint(model.hour, rule=ramp_down_rule)

def startup_logic(m, h):
    if h == 0:
        return m.v[h] >= m.u[h] -1 # assume plant initially off, if you want it on add -1 on the right
    return m.v[h] >= m.u[h] - m.u[h-1]
model.startup_logic = pyo.Constraint(model.hour, rule=startup_logic)

def total_startups_rule(m):
    return sum(m.v[h] for h in m.hour) <= max_startups # sum of startup flags limited by max_startups

model.max_startups_constraint = pyo.Constraint(rule=total_startups_rule)

def cumulative_uptime_rule(m):
    return sum(m.u[h] for h in m.hour) >= min_cumulative_uptime # sum of on flags for all hours is atleast min_cumulative_uptime
model.min_cumulative_uptime = pyo.Constraint(rule=cumulative_uptime_rule)

def cumulative_power_rule(m):
    return sum(m.p[h] for h in m.hour) >= min_cumulative_power # sum of on flags for all hours is atleast min_cumulative_uptime
model.min_cumulative_power = pyo.Constraint(rule=cumulative_power_rule)

# def shc(p):
#     return 1.38+0.000039*(p/max_power*100)**2-0.00778*(p/max_power*100)

# --- Objective: Maximize profit ---
def obj_rule(m):
    revenue = sum(m.p[t] * market_price[t] for t in T)
    gen_cost = sum((coal_price * heat_rate * degradation[t] + co2_price_bgn * emissions) * m.p[t] for t in T)
    startup_costs = sum(m.v[t] * startup_cost for t in T)
    return revenue - gen_cost - startup_costs
model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

# --- Solve ---
solver = pyo.SolverFactory('cbc', executable=r'C:\Program Files\Cbc-releases.2.10.12-w64-msvc17-md\bin\cbc.exe')
result = solver.solve(model, tee=True)

# --- Extract results ---
power = [pyo.value(model.p[t]) for t in T]
commitment = [pyo.value(model.u[t]) for t in T]
startups = [pyo.value(model.v[t]) for t in T]

revenue = sum(power[t] * market_price[t] for t in T)
gen_cost = sum((coal_price * heat_rate * degradation[t] + co2_price_bgn * emissions) * power[t] for t in T)
startup_total = sum(startups[t] * startup_cost for t in T)
total_profit = revenue - gen_cost - startup_total

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(T, market_price[:n_hours], label="Market Price (BGN/MWh)", color='black')
plt.step(T, power, label="Power Output (MW)", where='mid', linewidth=2)
plt.fill_between(T, 0, [max_power*u for u in commitment], color='lightgreen', alpha=0.3, step='mid', label="Committed")
plt.xlabel("Hour")
plt.ylabel("Value")
plt.title("Unit Commitment with Economic Dispatch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Total commitment: {sum(commitment)}h out of {n_hours} hours total")
print(f"Equals {(sum(commitment)/n_hours)*100:,.2f}% relative uptime")
print(f"Total revenue: BGN {revenue:,.2f}, per MWh: {revenue/sum(power):,.2f} BGN/MWh")
print(f"Total profit: BGN {total_profit:,.2f}, per MWh: {total_profit/sum(power):,.2f} BGN/MWh")
print(f"Total expenses: BGN {(gen_cost+startup_total):,.2f}, per MWh: {(gen_cost+startup_total)/sum(power):,.2f} BGN/MWh")
print(f"Total coal and CO2 expenses: BGN {gen_cost:,.2f}, per MWh: {gen_cost/sum(power):,.2f} BGN/MWh")
print(f"Total startups: {sum(startups):,.2f}, total cost of startups: BGN {(sum(startups)*startup_cost):,.2f}, per MWh: {(sum(startups)*startup_cost/sum(power)):,.2f} BGN/MWh")