from mytime import now
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def send_message(subject: str, body: str, filename: str = None):
    to_email = "yoavgilad555@gmail.com"
    from_email = "yoavgilad555@gmail.com"
    from_password = "hkxw clvy hpsy oenr"  # Use an App Password if 2FA is enabled

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(f'{now()}\n\n' + body, 'plain'))

    if filename is not None:
        # Attach the file
        with open(filename, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}',
            )
            msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Message sent successfully!")
    except Exception as e:
        print(f"Failed to send message: {e}")
