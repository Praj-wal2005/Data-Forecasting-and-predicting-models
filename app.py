from flask import Flask, render_template, request
import pandas as pd  # Required for reading CSV files
import forecast      # Imports your forecast.py logic

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    plot_html = None
    prediction_text = None
    
    if request.method == 'POST':
        # --- LOGIC 1: Check if a FILE was uploaded ---
        if 'user_file' in request.files:
            file = request.files['user_file']
            
            # Ensure the user actually selected a file
            if file.filename != '':
                try:
                    # Read the CSV file directly into pandas
                    df = pd.read_csv(file)
                    
                    # Call the function for FILES in forecast.py
                    plot_html, prediction_text = forecast.run_prediction_from_file(df)
                except Exception as e:
                    prediction_text = f"Error reading file: {e}"
        
        # --- LOGIC 2: Check if a TICKER was entered ---
        # We use elif so it doesn't try to do both at once
        elif request.form.get('user_input'):
             user_input = request.form.get('user_input')
             # Call the function for TICKERS in forecast.py
             plot_html, prediction_text = forecast.run_prediction(user_input)

    # Pass the results to the HTML page
    return render_template('index.html', plot_html=plot_html, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)