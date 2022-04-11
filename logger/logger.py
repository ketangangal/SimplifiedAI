from datetime import datetime


class Logger:
    def __init__(self, file="logger/logs/logs.log"):
        self.file = file

    def info(self, log_type, log_message):
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %H:%M:%S")
        with open(self.file, "a+") as file:
            file.write("[ " + current_time + " ~" + " Log_Type:" + log_type + " ] ~ " + log_message + "\n")
        file.close()