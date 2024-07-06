from flask import Blueprint, request, jsonify
from .qa_model import generate_qas

qa_bp = Blueprint('qa', __name__)

@qa_bp.route('/generate_qas', methods=['POST'])
def get_qas():
    data = request.get_json(force=True)
    text = data['text']
    
    result = generate_qas(text)
    return jsonify(result)
