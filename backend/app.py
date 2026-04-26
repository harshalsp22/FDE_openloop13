from flask import Flask
from flask_cors import CORS
from database import init_db
from routes.auth import auth_bp
from routes.patient import patient_bp
from routes.predict import predict_bp
from routes.image import image_bp

app = Flask(__name__)
app.secret_key = "fde_secret_key_2024"
CORS(app)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(patient_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(image_bp)

@app.route("/")
def home():
    return {"status": "FDE MedLogic API Running"}

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
