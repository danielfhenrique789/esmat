from sys_lib.context import Context

def get_routes(context):
    if not isinstance(context, Context):
        raise Exception('Is not a context object.')
    
    config = context.config
    utils = context.utils

    routes = []
    modules_path = "modules"
    modules = utils.get_folders_from_path(modules_path)
    for module in modules:
        if utils.file_exists(f"{modules_path}/{module}/controller.py"):
            route = {}
            _module = __import__(f"{modules_path}.{module}.controller", fromlist=["get_class"])
            route["class"] = _module.get_class(context)
            route["endpoint"] = f"/{config.flow_name.lower()}/{module.lower()}"
            routes.append(route)

    return routes