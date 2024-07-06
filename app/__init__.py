from flask import Flask

def create_app():
    app = Flask(__name__)
    from .routes import qa_bp
    app.register_blueprint(qa_bp)
    return app
