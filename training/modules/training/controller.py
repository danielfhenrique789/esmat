import requests
from flask_restful import abort 
import werkzeug
from modules.training.models.training import Training

def get_class(context):

    resource = context.resource
    parser = context.parser
    config = context.config

    def abort_if_todo_doesnt_exist(todo_id):
        if todo_id not in TODOS:
            abort(404, message="Todo {} doesn't exist".format(todo_id))

    # Todo
    # shows a single todo item and lets you delete a todo item
    class Controller(resource):
        def post(self):
            parser.add_argument('compilado', type=werkzeug.datastructures.FileStorage, location='files')
            parser.add_argument('esmat', type=werkzeug.datastructures.FileStorage, location='files')
            parser.add_argument('sag', type=werkzeug.datastructures.FileStorage, location='files')

            args = parser.parse_args()

            compiladoFile = args['compilado']
            esmatFile = args['esmat']
            sagFile = args['sag']
            
            compiladoFile.save("tmp/compilado_ARTS.xlsx")
            esmatFile.save("tmp/eSMAT_Goepik.xlsx")
            sagFile.save("tmp/SAG_Jandira.xlsx")

            # file = open('important', 'r')
            # requests.post("", )

            _model = Training(context).apply(), 200

            # open a file, where you ant to store the data
            pickle_path = "tmp/modelfile/" 
            file = open(pickle_path + config.flow_name, 'wb')

            # dump information to that file
            pickle.dump(_model, file)

            # close the file
            file.close()

            return {"response":"Training process successfully!"}, 200

    return Controller
