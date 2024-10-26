import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_gauge_chart(value1, title1, value2, title2):
    # Create a subplot figure with two columns
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'indicator'}, {'type': 'indicator'}]])

    # Add the first gauge chart
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = value1,
        title = {'text': title1},
        gauge = {'axis': {'range': [None, 5]}}
    ), row=1, col=1)

    # Add the second gauge chart
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = value2,
        title = {'text': title2},
        gauge = {'axis': {'range': [None, 100]}}
    ), row=1, col=2)

    # Update layout
    fig.update_layout(height=400, width=800)

    # Show figure
    fig.show()