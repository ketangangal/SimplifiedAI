import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader
import os


def template_loader(temp='email_template.html'):
    env = Environment(
        loader=FileSystemLoader('%s/' % os.path.dirname(__file__)))
    return env.get_template(temp)


def email_sender(receiver_email, logger):
    try:
        sender = "ketangangal98@gmail.com"
        my_password = 'dfdw yazr ckiy pjqu'
        receiver = receiver_email

        msg = MIMEMultipart('alternative')
        msg['Subject'] = "SIMPLIFIED AI : Process Completed"
        msg['From'] = sender
        msg['To'] = receiver

        html = template_loader()

        part2 = MIMEText(html.render(), 'html')

        msg.attach(part2)
        s = smtplib.SMTP_SSL('smtp.gmail.com')
        s.login(sender, my_password)

        s.sendmail(sender, receiver, msg.as_string())
        s.quit()
        return True
    except Exception as e:
        return e
