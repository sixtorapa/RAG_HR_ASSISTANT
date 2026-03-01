# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate

from config import Config

db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Extensiones
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    # Flask-Login config
    login_manager.login_view = "main.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"

    # User loader
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(user_id)

    # Filtro markdown para templates
    import markdown as md

    @app.template_filter('markdown')
    def markdown_filter(text):
        return md.markdown(text or '', extensions=['extra', 'nl2br'])

    # LangSmith (opcional, no rompe si no está configurado)
    from observability import init_langsmith
    init_langsmith()

    # Blueprint
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    return app

