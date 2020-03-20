import controller
from sys_lib.context import Context

def get_routes(context):
    if not isinstance(context, Context):
        raise Exception('Is not a context object.')
    
    config = context.config

    Class = controller.get_class(context)

    return [
        {"class":Class, "endpoint":config.endpoints["predict"]}]