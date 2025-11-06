from django import forms

class PlantParametersForm(forms.Form):
    min_power = forms.FloatField(initial=270, label="Minimum Power (MW)")
    max_power = forms.FloatField(initial=600, label="Maximum Power (MW)")
    ramp_up = forms.FloatField(initial=100, label="Ramp Up (MW/h)")
    ramp_down = forms.FloatField(initial=100, label="Ramp Down (MW/h)")
    emissions = forms.FloatField(initial=1.447, label="Emissions (t CO2/MWh)")
    offered_price = forms.FloatField(initial=312, label="Offered Price (BGN)")
    coal_price = forms.FloatField(initial=2.98e-6, label="Coal Price (BGN/kJ)")
    heat_rate = forms.FloatField(initial=10322e3, label="Heat Rate (kJ/MWh)")
    co2_price_bgn = forms.FloatField(initial=81.56 * 1.95583, label="CO2 Price (BGN/t)")
    startup_cost = forms.FloatField(initial=49191, label="Startup Cost (BGN)")
    max_startups = forms.IntegerField(initial=33, label="Max Startups")
    min_cumulative_uptime = forms.FloatField(initial=0, label="Min Uptime (h)")
    min_cumulative_power = forms.FloatField(initial=1858758, label="Min Cumulative Power (MWh)")

    file = forms.FileField(label="Excel Forecast File (.xlsx)")
