from website import create_app
import website.models

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)


