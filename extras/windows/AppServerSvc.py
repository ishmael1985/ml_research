import win32serviceutil
import win32service
import win32event
import servicemanager
import socket

import os
import json
import shutil
import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

os.chdir(os.path.dirname(os.path.realpath(__file__)))
 
def send_email_notification(service_params):
    fromaddr = service_params['email']['sender']
    recipients = service_params['email']['recipients']
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = service_params['email']['title']
 
    body = service_params['email']['body']
    msg.attach(MIMEText(body, 'plain'))
 
    server = smtplib.SMTP(service_params['email']['smtp_server'], 587)
    server.starttls()
    server.login(fromaddr, service_params['email']['password'])
    server.send_message(msg)
    server.quit()

def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()

class AppServerSvc (win32serviceutil.ServiceFramework):
    _svc_name_ = "MonitorResults"
    _svc_display_name_ = "Monitor Results"

    with open("service.json", "r") as service_config:
        service_params = json.load(service_config)

    results_file = service_params['results_worksheet']
    current_md5sum = ''

    def __init__(self,args):
        win32serviceutil.ServiceFramework.__init__(self,args)
        self.hWaitStop = win32event.CreateEvent(None,0,0,None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                          servicemanager.PYS_SERVICE_STARTED,
                          (self._svc_name_,''))
        self.main()

    def main(self):
        # Your business logic or call to any class should be here
        rc = None
        while rc != win32event.WAIT_OBJECT_0:
            new_md5sum = md5sum(self.results_file)
            if self.current_md5sum != new_md5sum:
                self.current_md5sum = new_md5sum
                shutil.copy2(self.results_file,
                             self.service_params['share_drive'])
                send_email_notification(self.service_params)
            # block for 24*60*60 seconds and wait for a stop event
            # it is used for a one-day loop
            rc = win32event.WaitForSingleObject(self.hWaitStop, 12 * 60 * 60 * 1000)

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(AppServerSvc)
