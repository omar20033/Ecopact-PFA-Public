from flask import Flask

import website.models


           


def create_app():
    app = Flask(__name__);
    app.config['SECRET_KEY'] = 'hgfhdfjdfhjgfh';
    
    

    from .views import views
    from .auth import auth
   


    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/auth')

    
    return app


