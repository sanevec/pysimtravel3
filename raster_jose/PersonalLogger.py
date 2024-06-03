import os
import datetime

class MyLogger:
    
    def __init__(self, save_path = "C://Users//josem//Jose//Sanevec//test_git//test_jose//log", name = "", log = True) -> None:
        self.log = log
        if log:
            today = datetime.date.today().strftime("%Y_%m_%d")

            if name:
                save_path+=f"//{name}//{today}"
            else:
                save_path+=f"common//{today}"
                
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            self.path_file = save_path + "//log_" + name + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            
            assert not os.path.exists(self.path_file), "El archivo ya existe"
                        
            with open(self.path_file, 'w') as file:
                file.write("------------ LOG INIT ----------\n")
                print(f"Se ha creado correctamente el log: {self.path_file}")
        else:
            print("No se va a hacer uso de log")
        
        
    def __call__(self, msg, lvl = "LOG"):
        
        if isinstance(msg, list):
            _msg = msg[0]
            for m in msg[1:]:
                _msg += "\n\t\t" + m
            msg = _msg  

        if not self.log:
            print(msg)
            return
        with open(self.path_file, 'a') as file:
            file.write(f"{lvl}- {self.get_time}: {msg}\n")
            
    @property
    def get_time(self):
        return f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"