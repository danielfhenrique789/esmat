import requests
from flask_restful import abort 
import werkzeug
from modules.predict.models.predict import Predict

def get_class(context):

    resource = context.resource
    parser = context.parser

    def abort_if_todo_doesnt_exist(todo_id):
        if todo_id not in TODOS:
            abort(404, message="Todo {} doesn't exist".format(todo_id))

    # Todo
    # shows a single todo item and lets you delete a todo item
    class Controller(resource):
        def post(self):
            return Predict(context).apply(), 200

        
    return Controller
