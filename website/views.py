import base64
import io
from flask import Flask, render_template, Blueprint,url_for,redirect,request
from .auth import auth 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import website.models

views=Blueprint('views',__name__,template_folder='../Templates')



@views.route("/")
def home() :
    return redirect(url_for('auth.login'))
@views.route('/test')
def plot():
    # Sample data
    user_id = 'ali.frihida@enit.utm.tn'
    components = {
        'Date': [component[2] for component in website.models.UserComponents(user_id)],
        'Value': [component[1] for component in website.models.UserComponents(user_id)],
        'Component': [component[0] for component in website.models.UserComponents(user_id)]
    }

    # Create DataFrame
    df = pd.DataFrame(components)

    # Convert 'Date' column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as index
    df.set_index('Date', inplace=True)

    # Filter data for components 'P', 'NA', 'Ph'
    df_components = {}
    for component in ['P', 'NA', 'Ph']:
        df_component = df[df['Component'] == component]
        df_components[component] = df_component[~df_component.index.duplicated()].resample('M').ffill()

    # Plotting
    plots = []
    for component, df_component in df_components.items():
        plt.figure(figsize=(10, 6))
        plt.plot(df_component.index, df_component['Value'], marker='o', linestyle='-')
        plt.title(f'Component {component} - Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)

        # Format date on x-axis
        date_form = DateFormatter("%Y-%m-%d")  # Customize the date format as per your preference
        plt.gca().xaxis.set_major_formatter(date_form)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # Save the plot as a PNG image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        plots.append({'component': component, 'image_base64': image_base64})

    # Render template with plots
    return render_template('test1.html', plots=plots)






