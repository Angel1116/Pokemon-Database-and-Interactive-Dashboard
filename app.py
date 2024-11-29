app = dash.Dash(__name__)

# For deployment
server = app.server


if __name__ == '__main__':
    app.run_server(debug=True)
    