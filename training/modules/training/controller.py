import requests
import sys
from flask_restful import abort 
import werkzeug
from modules.training.models.training import Training
from modules.training.models.esmat_files import EsmatFiles
from modules.training.config import config_training
import pickle

def get_class(context):

    resource = context.resource
    parser = context.parser
    config = context.config
    logging = context.logging
    esmat_files = EsmatFiles(config_training)

    def abort_if_todo_doesnt_exist(todo_id):
        if todo_id not in TODOS:
            abort(404, message="Todo {} doesn't exist".format(todo_id))

    # Todo
    # shows a single todo item and lets you delete a todo item
    class Controller(resource):
        def get(self):
            return {"test": "Didier"}
        def post(self):
            try:
                parser.add_argument('compilado', type=werkzeug.datastructures.FileStorage, location='files')
                parser.add_argument('esmat', type=werkzeug.datastructures.FileStorage, location='files')
                parser.add_argument('sag', type=werkzeug.datastructures.FileStorage, location='files')

                args = parser.parse_args()

                compiladoFile = args['compilado']
                esmatFile = args['esmat']
                sagFile = args['sag']
                
                compiladoFile.save(esmat_files.compilado)
                esmatFile.save(esmat_files.esmat)
                sagFile.save(esmat_files.sag)

                _model = Training(context, esmat_files).apply(), 200

                # open a file, where you ant to store the data
                pickle_path = "tmp/modelfile/" 
                file = open(pickle_path + config.flow_name, 'wb')

                # dump information to that file
                pickle.dump(_model, file)

                # close the file
                file.close()
                msg_success = "Training process successfully!"
                logging.info(msg_success)
                status_code = 200
                response = {"response: ": msg_success}
            except OSError as err:
                response = {"response: ": "OS error: {0}".format(err)}
                logging.error(response)
            except:
                status_code = 503
                response = {"response: ": "OS error: {0}".format(err)}
                logging.error(response)
            finally:
                return response, status_code

    return Controller
