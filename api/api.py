from flask import Flask, request, Response
from flask_restful import Api, Resource, reqparse
from model import predict_model
import numpy as np
import random
import json

model = predict_model()

app = Flask(__name__)
api = Api(app)

class PredictApiProcessor(Resource):
    def get(self):
        return "Воспользуйтесь методом POST для передачи данных ЭКГ. Входящий массив должен иметь вид [[0,0,0], [0,0,0], ... ]", 201
    
    def post(self):
        data = request.get_json(force=True)

        success, data, error = model.predict(data['array'])
        if not success:
            return error, 409
        else:
            res = json.dumps({"results": data.tolist()})
            return Response(res, content_type="application/json")
        

api.add_resource(PredictApiProcessor, "/")

app.run(port=5000)