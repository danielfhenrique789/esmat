class EsmatFiles:

    def __init__(self, config):
        files_folder = config["files_folder"]
        self.compilado = files_folder + config["compilado_file"]
        self.esmat = files_folder + config["esmat_file"]
        self.sag = files_folder + config["sag_file"]
