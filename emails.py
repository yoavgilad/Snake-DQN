from mytime import now  # own module
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os


def send_message(subject: str, body: str, filenames: list[str] = None):
    # Define sender and recipient
    receiver_email = "example@gmail.com"
    sender_email = "example@gmail.com"
    sender_password = "xxxx xxxx xxxx xxxx"  # Use an App Password if 2FA is enabled
    # Define email message object
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    # Attach body
    msg.attach(MIMEText(f'{now()}\n\n' + body, 'plain'))
    # Attach files if given
    if filenames is not None:
        for filename in filenames:
            with open(filename, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(filename)}',
                )
                msg.attach(part)
    # Send email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Message sent successfully!")
    except Exception as e:
        print(f"Failed to send message: {e}")
