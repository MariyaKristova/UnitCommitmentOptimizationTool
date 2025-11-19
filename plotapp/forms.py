import os
import pandas as pd
from django import forms
from django.conf import settings


class PlantParametersForm(forms.Form):
    excel_file = forms.ChoiceField(label="SELECT EXCEL FILE")
    min_power = forms.FloatField(initial=270, label="Minimum Power (MW)")
    max_power = forms.FloatField(initial=600, label="Maximum Power (MW)")
    ramp_up = forms.FloatField(initial=100, label="Ramp Up (MW/h)")
    ramp_down = forms.FloatField(initial=100, label="Ramp Down (MW/h)")
    emissions = forms.FloatField(initial=1.447, label="Emissions (t CO2/MWh)")
    # offered price field is not being used for now
    # offered_price = forms.FloatField(initial=312, label="Offered Price (BGN)")
    coal_price = forms.FloatField(initial=2.98e-6, label="Coal Price (BGN/kJ)")
    heat_rate = forms.FloatField(initial=10322e3, label="Heat Rate (kJ/MWh)")
    co2_price_bgn = forms.FloatField(initial=81.56 * 1.95583, label="CO2 Price (BGN/t)")
    startup_cost = forms.FloatField(initial=49191, label="Startup Cost (BGN)")
    max_startups = forms.IntegerField(initial=33, label="Max Startups")
    min_cumulative_uptime = forms.FloatField(initial=0, label="Min Uptime (h)")
    min_cumulative_power = forms.FloatField(initial=1858758, label="Min Cumulative Power (MWh)")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_dir = settings.DATA_INPUT_DIR
        files = []
        if os.path.exists(data_dir):
            files = [
                f for f in os.listdir(data_dir)
                if f.lower().endswith((".xlsx", ".xls"))
            ]

        self.fields["excel_file"].choices = [(f, f) for f in files]

    def clean_excel_file(self):
        excel_filename = self.cleaned_data.get('excel_file')
        excel_path = os.path.join(settings.DATA_INPUT_DIR, excel_filename)

        if not os.path.exists(excel_path):
            raise forms.ValidationError(f"Excel file {excel_filename} not found on the server.")

        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            raise forms.ValidationError(f"Error reading Excel file: {e}")

        if 'DateTime' not in df.columns or 'Price' not in df.columns:
            raise forms.ValidationError("Excel file must contain 'DateTime' and 'Price' columns")

        n_hours = df.shape[0]
        if n_hours not in (365 * 24, 366 * 24):
            raise forms.ValidationError(
                f"Excel file must contain a full year of hourly data (8760 or 8784 rows), got {n_hours}"
            )

        return excel_filename


class ExtractPeriodForm(forms.Form):
    start_date = forms.DateField(
        input_formats=['%d.%m'],
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'DD.MM'})
    )
    end_date = forms.DateField(
        input_formats=['%d.%m'],
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'DD.MM'})
    )

    def clean(self):
        cleaned = super().clean()
        start = cleaned.get('start_date')
        end = cleaned.get('end_date')

        # If one of the fields failed basic validation, stop here
        if not start or not end:
            return cleaned

        # Prevent cross-year intervals (e.g., 31.12 → 01.01)
        if start.month == 12 and end.month == 1:
            raise forms.ValidationError(
                "The selected date range cannot cross the year boundary (31.12 → 01.01 is not allowed)."
            )

        # End date must not be earlier than start date
        if end < start:
            raise forms.ValidationError(
                "End date must be later than start date."
            )

        return cleaned



