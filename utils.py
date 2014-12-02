import smtplib

from email.MIMEImage import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage

from config import *

def send_mail(text, filename=''):
    global email_username, email_password
    fromaddr = 'hexa.portal@gmail.com'

    recipients = ["carpedm20@gmail.com"]
    toaddrs  = ", ".join(recipients)

    username = email_username
    password = email_password

    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = text
    msgRoot['From'] = fromaddr
    msgRoot['To'] = toaddrs

    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)

    msgText = MIMEText("Finished", 'html')
    msgAlternative.attach(msgText)

    if filename is not '':
      img = MIMEImage(open(filename,"rb").read(), _subtype="png")
      img.add_header('Content-ID', '<carpedm20>')
      msgRoot.attach(img)
      
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, recipients, msgRoot.as_string())
    server.quit()
    print " - mail sended"
