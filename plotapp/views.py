from io import BytesIO
import base64

from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
import matplotlib.pyplot as plt


def upload_view(request):
    # Check if the request method is POST (user submitted the form)
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)

        # Validate the form (check that a file is uploaded)
        if form.is_valid():
            excel_file = request.FILES['file']

            # Try to read the uploaded Excel file into a Pandas DataFrame
            try:
                df = pd.read_excel(excel_file)
            except Exception as e:
                # If reading the file fails, show an error message
                return render(request, 'plotapp/upload.html', {
                    'form': form,
                    'error': f'Error reading Excel file: {e}'
                })

            # Select numeric columns only (required for plotting)
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) == 0:
                # No numeric data — cannot plot
                return render(request, 'plotapp/upload.html', {
                    'form': form,
                    'error': 'No numeric columns found in the uploaded file.'
                })

            # Use the first numeric column as Y-axis data
            y_col = num_cols[0]
            y = df[y_col].dropna().values  # Remove NaN values
            x = range(len(y))              # X-axis as sequential index

            # Plot the curve in memory
            plt.figure(figsize=(10, 4))
            plt.plot(x, y, linewidth=2)
            plt.xlabel('Index')
            plt.ylabel(str(y_col))
            plt.title('Curve from Excel Data')
            plt.grid(True)
            plt.tight_layout()

            # Save plot to a memory buffer as PNG
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()     # Free memory by closing the figure
            buf.seek(0)     # Go to start of buffer

            # Convert PNG image to base64 string for HTML embedding
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Render the result page with the encoded image
            return render(request, 'plotapp/result.html', {
                'image': image_base64,
                'col_name': str(y_col)
            })

    else:
        # If GET request — show empty upload form
        form = UploadFileForm()

    # Render the upload page (for GET or if errors occurred)
    return render(request, 'plotapp/upload.html', {'form': form})
